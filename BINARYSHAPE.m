classdef BINARYSHAPE < PROBLEM
% <multi> <binary> <none/none>
% Binary-encoded geometric shape distance problem
%
% Parameters (Problem.Parameter):
%   shapeType  : shape selector
%                1 = rectangle (4 objectives)
%                2 = circle / annulus (2 objectives)
%                3 = triangle (3 objectives)
%                4 = regular pentagon (5 objectives)
%                5 = regular k-gon (k objectives)
%   kPoly      : number of edges when shapeType = 5 (k-gon), default 6
%   bitsPerDim : number of bits per coordinate dimension, default 10
%
% Decision variables:
%   Binary string of length D = 2 * bitsPerDim
%   First bitsPerDim bits encode x1 in [0,1]
%   Last  bitsPerDim bits encode x2 in [0,1]
%
% Objectives:
%   - For rectangle: f1..f4 = distances to four axis-parallel lines
%   - For circle   : f1,f2   = distance to inner and outer circles
%   - For triangle : f1..f3 = distances to three edges
%   - For polygon  : f1..fk = distances to each polygon edge
%
% All objectives are minimization (distance-like).
%
% NEW (analysis tools):
%   - DrawClustered   : visualize population in (x1,x2) with colors
%                       determined by clusters in binary (Hamming) space.
%   - ClusterAnalysis : perform Hamming-space clustering and return
%                       cluster IDs and bit-frequency profiles.
%   - AnalyzeClusters : one-shot analysis wrapper that runs clustering
%                       and prints human-readable descriptions of each
%                       cluster's bit-pattern characteristics.

    properties(Access = private)
        shapeType   % 1=rect, 2=circle, 3=triangle, 4=pentagon, 5=k-gon
        kPoly       % number of edges when shapeType=5
        bitsPerDim  % bits per coordinate (x1,x2)
        % Precomputed geometric parameters
        rect_a1
        rect_a2
        rect_b1
        rect_b2
        circle_center
        circle_R1
        circle_R2
        poly_vertices   % 2 x K matrix of polygon vertices (for tri/penta/k-gon)
        poly_normals    % 2 x K matrix of outward normals for polygon edges
        poly_d          % 1 x K scalars: n_i' * v_i (line offsets)
    end

    methods
        %% Default settings of the problem
        function Setting(obj)
            % ----- default M and D will be set after reading parameters -----
            if isempty(obj.M)
                obj.M = 4; % temporary, will be overwritten
            end
            if isempty(obj.D)
                obj.D = 20; % temporary, will be overwritten
            end

            % Read parameters:
            % [shapeType, kPoly, bitsPerDim]
            [shapeType, kPoly, bitsPerDim] = obj.ParameterSet(5, 6, 20);

            % Sanitize
            shapeType  = max(1,min(5,round(shapeType)));
            kPoly      = max(3,round(kPoly));        % at least triangle
            bitsPerDim = max(2,round(bitsPerDim));   % at least 2 bits

            obj.shapeType  = shapeType;
            obj.kPoly      = kPoly;
            obj.bitsPerDim = bitsPerDim;

            % Set number of objectives according to shapeType
            switch shapeType
                case 1 % rectangle
                    obj.M = 4;
                case 2 % circle / annulus
                    obj.M = 2;
                case 3 % triangle
                    obj.M = 3;
                case 4 % pentagon
                    obj.M = 5;
                case 5 % k-gon
                    obj.M = kPoly;
            end

            % Decision length: two coordinates, each bitsPerDim bits
            obj.D        = 2*bitsPerDim;
            obj.encoding = 'binary';

            % Binary variables: lower=0, upper=1
            obj.lower    = zeros(1,obj.D);
            obj.upper    = ones(1,obj.D);

            % ----- geometric parameters in [0,1]x[0,1] -----
            % 1) Rectangle: inner box [a1,a2]x[b1,b2]
            obj.rect_a1 = 0.25;
            obj.rect_a2 = 0.75;
            obj.rect_b1 = 0.25;
            obj.rect_b2 = 0.75;

            % 2) Circle: center + inner/outer radius
            obj.circle_center = [0.5; 0.5];
            obj.circle_R1     = 0.20;
            obj.circle_R2     = 0.35;

            % 3/4/5) Polygon (triangle/pentagon/k-gon)
            switch shapeType
                case 3
                    K = 3;
                    % Simple triangle in the middle of [0,1]^2
                    V = [0.2 0.8 0.5;
                         0.2 0.2 0.8];   % 2 x 3
                case 4
                    K = 5;
                    % Regular pentagon centered at (0.5,0.5)
                    [V] = regularPolygonVertices(0.5,0.5,0.30,K, -pi/2);
                case 5
                    K = kPoly;
                    [V] = regularPolygonVertices(0.5,0.5,0.30,K, -pi/2);
                otherwise
                    K = 0;
                    V = [];
            end

            if ~isempty(V)
                [N, dvec] = polygonNormals(V);
                obj.poly_vertices = V;
                obj.poly_normals  = N;
                obj.poly_d        = dvec;
            else
                obj.poly_vertices = [];
                obj.poly_normals  = [];
                obj.poly_d        = [];
            end
        end

        %% Calculate objective values
        function PopObj = CalObj(obj,PopDec)
            % PopDec: N x D, binary
            % Decode to continuous X in [0,1]^2
            X = decodeBinaryToXY(obj,PopDec); % N x 2

            N = size(X,1);
            PopObj = zeros(N,obj.M);

            switch obj.shapeType
                case 1  % rectangle
                    x1 = X(:,1);
                    x2 = X(:,2);
                    f1 = abs(x1 - obj.rect_a1);
                    f2 = abs(x1 - obj.rect_a2);
                    f3 = abs(x2 - obj.rect_b1);
                    f4 = abs(x2 - obj.rect_b2);
                    PopObj(:,1) = f1;
                    PopObj(:,2) = f2;
                    PopObj(:,3) = f3;
                    PopObj(:,4) = f4;

                case 2  % circle / annulus
                    C  = obj.circle_center(:)'; % 1 x 2
                    dx = X(:,1) - C(:,1);
                    dy = X(:,2) - C(:,2);
                    r  = sqrt(dx.^2 + dy.^2);
                    PopObj(:,1) = abs(r - obj.circle_R1);
                    PopObj(:,2) = abs(r - obj.circle_R2);

                case {3,4,5} % triangle / pentagon / k-gon
                    V    = obj.poly_vertices;   % 2 x K
                    Nrm  = obj.poly_normals;    % 2 x K
                    dvec = obj.poly_d;          % 1 x K
                    K    = size(V,2);

                    for k = 1:K
                        n  = Nrm(:,k);   % 2x1
                        d0 = dvec(k);    % scalar
                        % distance from X to line: |n' * x - d| / ||n||
                        num = abs(X * n - d0);
                        den = norm(n);
                        PopObj(:,k) = num / den;
                    end
            end
        end

        %% Generate a point for hypervolume calculation (rough heuristic)
        function R = GetOptimum(obj,~)
            % All objectives are distances in [0, ~1], so a simple
            % conservative reference point at 1.5 for each dimension
            R = 1.5 * ones(1,obj.M);
        end

        %% Display a population in the decision space (shape visualization)
        function DrawObj(obj,Population)
            % Decode to (x1,x2) and scatter; ignore objective space here
            X = decodeBinaryToXY(obj,Population.decs);
            scatter(X(:,1),X(:,2),15,'filled');
            xlim([0,1]); ylim([0,1]);
            xlabel('x_1'); ylabel('x_2'); box on; axis equal;
            title(['BINARYSHAPE, type = ',num2str(obj.shapeType)]);
            hold on;

            % Optionally draw the target shape for intuition
            switch obj.shapeType
                case 1 % rectangle
                    rectangle('Position',[obj.rect_a1,obj.rect_b1,...
                        obj.rect_a2-obj.rect_a1,obj.rect_b2-obj.rect_b1], ...
                        'EdgeColor','r','LineWidth',1.5);
                case 2 % circle (draw outer + inner)
                    th = linspace(0,2*pi,200);
                    cx = obj.circle_center(1) + obj.circle_R2*cos(th);
                    cy = obj.circle_center(2) + obj.circle_R2*sin(th);
                    plot(cx,cy,'r-','LineWidth',1.5);
                    cx1 = obj.circle_center(1) + obj.circle_R1*cos(th);
                    cy1 = obj.circle_center(2) + obj.circle_R1*sin(th);
                    plot(cx1,cy1,'r-','LineWidth',1.5);
                case {3,4,5}
                    if ~isempty(obj.poly_vertices)
                        V  = obj.poly_vertices;
                        Vc = [V, V(:,1)]; % close polygon
                        plot(Vc(1,:),Vc(2,:),'r-','LineWidth',1.5);
                    end
            end
            hold off;
        end

        %% ---- NEW: clustered visualization in (x1,x2) with 01-space clusters ----
        function DrawClustered(obj,Population,K)
            % DrawClustered
            % Visualize the population in (x1,x2) space, color-coded by
            % clusters obtained in the binary (Hamming) space.
            %
            % Usage:
            %   problem.DrawClustered(Population);       % default K
            %   problem.DrawClustered(Population, K);    % user-defined K

            if nargin < 3 || isempty(K)
                % default: up to 6 clusters, but not more than population size
                K = min(6, size(Population.decs,1));
            end

            Z = Population.decs;           % N x D, 0/1
            N = size(Z,1);
            if N == 0
                warning('DrawClustered: Empty population.');
                return;
            end

            % Decode to (x1,x2)
            X = decodeBinaryToXY(obj,Z);   % N x 2

            % Hamming-space clustering (k-medoids on precomputed distances)
            Zdouble = double(Z);
            K = min(K, N);                 % cannot have more clusters than points
            if K <= 1
                clusterID = ones(N,1);
            else
                try
                    % Pairwise Hamming distances
                    Dmat = pdist2(Zdouble,Zdouble,'hamming');
                    opts = statset('MaxIter',500,'Display','off');
                    [clusterID,~] = kmedoids(Dmat,K, ...
                        'Distance','precomputed', ...
                        'Options',opts,'Replicates',5);
                catch
                    % Fallback: k-means with cityblock distance
                    warning('kmedoids or pdist2 not available, falling back to kmeans(cityblock).');
                    opts = statset('MaxIter',500,'Display','off');
                    [clusterID,~] = kmeans(Zdouble,K, ...
                        'Distance','cityblock', ...
                        'Replicates',5,'Options',opts);
                end
            end

            % Plot in (x1,x2) and color by cluster
            figure('Color','w');
            hold on; box on; axis equal;
            xlim([0,1]); ylim([0,1]);
            xlabel('x_1'); ylabel('x_2');
            title(['BINARYSHAPE clustered view, type = ',num2str(obj.shapeType)]);

            cmap = lines(K);
            for k = 1:K
                idx = (clusterID == k);
                if any(idx)
                    scatter(X(idx,1),X(idx,2),28, ...
                        'MarkerFaceColor',cmap(k,:), ...
                        'MarkerEdgeColor','none', ...
                        'MarkerFaceAlpha',0.7);
                end
            end

            % Draw the enclosing shape on top
            switch obj.shapeType
                case 1 % rectangle
                    rectangle('Position',[obj.rect_a1,obj.rect_b1,...
                        obj.rect_a2-obj.rect_a1,obj.rect_b2-obj.rect_b1], ...
                        'EdgeColor','k','LineWidth',1.5);
                case 2 % circle/annulus
                    th = linspace(0,2*pi,200);
                    cx = obj.circle_center(1) + obj.circle_R2*cos(th);
                    cy = obj.circle_center(2) + obj.circle_R2*sin(th);
                    plot(cx,cy,'k-','LineWidth',1.5);
                    cx1 = obj.circle_center(1) + obj.circle_R1*cos(th);
                    cy1 = obj.circle_center(2) + obj.circle_R1*sin(th);
                    plot(cx1,cy1,'k-','LineWidth',1.2);
                case {3,4,5}
                    if ~isempty(obj.poly_vertices)
                        V  = obj.poly_vertices;
                        Vc = [V, V(:,1)];
                        plot(Vc(1,:),Vc(2,:),'k-','LineWidth',1.5);
                    end
            end
            legendStrings = arrayfun(@(k)sprintf('Cluster %d',k),1:K,'UniformOutput',false);
            legend(legendStrings,'Location','bestoutside');
            hold off;
        end

        %% ---- NEW: cluster analysis helper (returns bit-level profiles) ----
        function info = ClusterAnalysis(obj,PopDec,K)
            % ClusterAnalysis
            % Perform Hamming-space clustering on binary decisions and
            % return cluster assignments and bit-frequency profiles.
            %
            % Usage:
            %   info = problem.ClusterAnalysis(PopDec);
            %   info = problem.ClusterAnalysis(PopDec,K);
            %
            % info has fields:
            %   .clusterID  : N x 1 cluster indices
            %   .K          : number of clusters used
            %   .bitFreq    : K x D matrix, bitFreq(k,j) in [0,1]
            %   .medoidIdx  : 1 x K indices of representative solutions (if available)

            Z = PopDec;
            N = size(Z,1);
            D = size(Z,2);

            if nargin < 3 || isempty(K)
                K = min(6,N);
            end
            K = min(K,N);

            info = struct('clusterID',[],'K',K,'bitFreq',[],'medoidIdx',[]);

            if N == 0
                warning('ClusterAnalysis: empty PopDec.');
                return;
            end

            Zdouble = double(Z);

            if K <= 1
                clusterID = ones(N,1);
                medoidIdx = 1;
            else
                try
                    Dmat = pdist2(Zdouble,Zdouble,'hamming');
                    opts = statset('MaxIter',500,'Display','off');
                    [clusterID,medoidIdx] = kmedoids(Dmat,K, ...
                        'Distance','precomputed', ...
                        'Options',opts,'Replicates',5);
                catch
                    warning('kmedoids or pdist2 not available, falling back to kmeans(cityblock).');
                    opts = statset('MaxIter',500,'Display','off');
                    [clusterID,~] = kmeans(Zdouble,K, ...
                        'Distance','cityblock', ...
                        'Replicates',5,'Options',opts);
                    % use nearest point to each centroid as pseudo-medoid
                    medoidIdx = zeros(1,K);
                    for k = 1:K
                        idx = find(clusterID == k);
                        if isempty(idx)
                            medoidIdx(k) = 1;
                        else
                            medoidIdx(k) = idx(1);
                        end
                    end
                end
            end

            % bit-frequency per cluster
            bitFreq = zeros(K,D);
            for k = 1:K
                idx = (clusterID == k);
                if any(idx)
                    Zk = Zdouble(idx,:);
                    bitFreq(k,:) = mean(Zk,1);
                else
                    bitFreq(k,:) = NaN;
                end
            end

            info.clusterID = clusterID(:);
            info.bitFreq   = bitFreq;
            info.medoidIdx = medoidIdx;
        end

        %% ---- NEW: one-shot cluster analysis + textual description ----
        function info = AnalyzeClusters(obj,Population,K,tauHigh,tauLow)
            % AnalyzeClusters
            %   Convenience wrapper that:
            %   (1) extracts binary decisions from Population,
            %   (2) runs ClusterAnalysis in Hamming space,
            %   (3) prints interpretable descriptions of each cluster's
            %       bit-pattern characteristics (which bits are almost
            %       always 1 or 0 in that cluster).
            %
            % Usage:
            %   info = problem.AnalyzeClusters(Population);
            %   info = problem.AnalyzeClusters(Population,K);
            %   info = problem.AnalyzeClusters(Population,K,tauHigh,tauLow);

            if nargin < 3 || isempty(K)
                K = min(6,size(Population.decs,1));
            end
            if nargin < 4 || isempty(tauHigh)
                tauHigh = 0.8;
            end
            if nargin < 5 || isempty(tauLow)
                tauLow  = 0.2;
            end

            Z = Population.decs;
            info = obj.ClusterAnalysis(Z,K);

            bitFreq = info.bitFreq;      % K x D
            K       = info.K;
            [K2,D]  = size(bitFreq);
            if K2 ~= K
                warning('AnalyzeClusters: inconsistent K in bitFreq.');
            end

            b = obj.bitsPerDim;
            if D ~= 2*b
                warning('AnalyzeClusters assumes D = 2 * bitsPerDim.');
            end

            fprintf('=== Cluster-based binary pattern summary ===\n');
            fprintf('K = %d clusters, D = %d bits (b = %d bits per dimension)\n',K,D,b);
            fprintf('Strong-1 if p >= %.2f, Strong-0 if p <= %.2f\n\n',tauHigh,tauLow);

            for k = 1:K
                pk = bitFreq(k,:);   % 1 x D
                if all(isnan(pk))
                    fprintf('Cluster %d: (empty)\n\n',k);
                    continue;
                end

                % strong-1 and strong-0 indices
                idxStrong1 = find(pk >= tauHigh);
                idxStrong0 = find(pk <= tauLow);

                % split into x1 and x2 parts
                idx1_1 = idxStrong1(idxStrong1 <= b);
                idx1_0 = idxStrong0(idxStrong0 <= b);
                idx2_1 = idxStrong1(idxStrong1 >  b) - b;
                idx2_0 = idxStrong0(idxStrong0 >  b) - b;

                % group consecutive indices into ranges for compact printing
                ranges1_1 = idx2ranges(idx1_1);
                ranges1_0 = idx2ranges(idx1_0);
                ranges2_1 = idx2ranges(idx2_1);
                ranges2_0 = idx2ranges(idx2_0);

                fprintf('Cluster %d:\n',k);

                % number of solutions in this cluster
                if isfield(info,'clusterID')
                    n_k = sum(info.clusterID == k);
                    fprintf('  Size: %d solutions\n', n_k);
                end

                % x1 part
                if ~isempty(ranges1_1)
                    fprintf('  x1 bits strongly 1 at: %s\n', ranges2str(ranges1_1));
                else
                    fprintf('  x1 bits strongly 1 at: (none)\n');
                end
                if ~isempty(ranges1_0)
                    fprintf('  x1 bits strongly 0 at: %s\n', ranges2str(ranges1_0));
                else
                    fprintf('  x1 bits strongly 0 at: (none)\n');
                end

                % x2 part
                if ~isempty(ranges2_1)
                    fprintf('  x2 bits strongly 1 at: %s\n', ranges2str(ranges2_1));
                else
                    fprintf('  x2 bits strongly 1 at: (none)\n');
                end
                if ~isempty(ranges2_0)
                    fprintf('  x2 bits strongly 0 at: %s\n', ranges2str(ranges2_0));
                else
                    fprintf('  x2 bits strongly 0 at: (none)\n');
                end

                fprintf('\n');
            end
        end
                %% ---- NEW: one-shot draw + analyze with consistent clustering ----
        function info = DrawAndAnalyzeClusters(obj,Population,K,tauHigh,tauLow)
            % DrawAndAnalyzeClusters
            %   (1) 从 Population 中提取二进制决策向量
            %   (2) 在 Hamming 空间做一次聚类（调用 ClusterAnalysis）
            %   (3) 在 (x1,x2) 空间按 clusterID 画图
            %   (4) 直接基于这一次聚类结果输出 bit 模式分析
            %
            % 使用方式：
            %   info = problem.DrawAndAnalyzeClusters(Population);
            %   info = problem.DrawAndAnalyzeClusters(Population,K);
            %   info = problem.DrawAndAnalyzeClusters(Population,K,tauHigh,tauLow);

            % ----- 参数默认值 -----
            if nargin < 3 || isempty(K)
                K = min(6,size(Population.decs,1));
            end
            if nargin < 4 || isempty(tauHigh)
                tauHigh = 0.8;
            end
            if nargin < 5 || isempty(tauLow)
                tauLow  = 0.2;
            end

            % ----- 提取二进制决策 -----
            Z = Population.decs;      % N x D, 0/1
            N = size(Z,1);
            if N == 0
                warning('DrawAndAnalyzeClusters: Empty population.');
                info = struct();
                return;
            end

            % ----- 一次性聚类（重用 ClusterAnalysis） -----
            info = obj.ClusterAnalysis(Z,K);   % 只在这里聚类一次
            clusterID = info.clusterID(:);
            K         = info.K;

            % ----- 解码到 (x1,x2) 并画图 -----
            X = decodeBinaryToXY(obj,Z);   % N x 2

            figure('Color','w');
            hold on; box on; axis equal;
            xlim([0,1]); ylim([0,1]);
            xlabel('x_1'); ylabel('x_2');
            title(['BINARYSHAPE clustered view, type = ',num2str(obj.shapeType)]);

            cmap = lines(K);
            for k = 1:K
                idx = (clusterID == k);
                if any(idx)
                    scatter(X(idx,1),X(idx,2),28, ...
                        'MarkerFaceColor',cmap(k,:), ...
                        'MarkerEdgeColor','none', ...
                        'MarkerFaceAlpha',0.7);
                end
            end

            % 画出几何形状轮廓
            switch obj.shapeType
                case 1 % rectangle
                    rectangle('Position',[obj.rect_a1,obj.rect_b1,...
                        obj.rect_a2-obj.rect_a1,obj.rect_b2-obj.rect_b1], ...
                        'EdgeColor','k','LineWidth',1.5);
                case 2 % circle/annulus
                    th = linspace(0,2*pi,200);
                    cx = obj.circle_center(1) + obj.circle_R2*cos(th);
                    cy = obj.circle_center(2) + obj.circle_R2*sin(th);
                    plot(cx,cy,'k-','LineWidth',1.5);
                    cx1 = obj.circle_center(1) + obj.circle_R1*cos(th);
                    cy1 = obj.circle_center(2) + obj.circle_R1*sin(th);
                    plot(cx1,cy1,'k-','LineWidth',1.2);
                case {3,4,5}
                    if ~isempty(obj.poly_vertices)
                        V  = obj.poly_vertices;
                        Vc = [V, V(:,1)];
                        plot(Vc(1,:),Vc(2,:),'k-','LineWidth',1.5);
                    end
            end

            legendStrings = arrayfun(@(k)sprintf('Cluster %d',k),1:K,'UniformOutput',false);
            legend(legendStrings,'Location','bestoutside');
            hold off;

            % ----- 使用同一 info 做 bit 模式分析并打印 -----
            bitFreq = info.bitFreq;      % K x D
            [K2,D]  = size(bitFreq);
            if K2 ~= K
                warning('DrawAndAnalyzeClusters: inconsistent K in bitFreq.');
            end

            b = obj.bitsPerDim;
            if D ~= 2*b
                warning('DrawAndAnalyzeClusters assumes D = 2 * bitsPerDim.');
            end

            fprintf('=== Cluster-based binary pattern summary (same clustering as the plot) ===\n');
            fprintf('K = %d clusters, D = %d bits (b = %d bits per dimension)\n',K,D,b);
            fprintf('Strong-1 if p >= %.2f, Strong-0 if p <= %.2f\n\n',tauHigh,tauLow);

            for k = 1:K
                pk = bitFreq(k,:);   % 1 x D
                if all(isnan(pk))
                    fprintf('Cluster %d: (empty)\n\n',k);
                    continue;
                end

                % strong-1 和 strong-0 的 bit 下标
                idxStrong1 = find(pk >= tauHigh);
                idxStrong0 = find(pk <= tauLow);

                % 划分到 x1 / x2
                idx1_1 = idxStrong1(idxStrong1 <= b);
                idx1_0 = idxStrong0(idxStrong0 <= b);
                idx2_1 = idxStrong1(idxStrong1 >  b) - b;
                idx2_0 = idxStrong0(idxStrong0 >  b) - b;

                % 连续下标合并成区间，方便打印（重用你已有的 idx2ranges / ranges2str）
                ranges1_1 = idx2ranges(idx1_1);
                ranges1_0 = idx2ranges(idx1_0);
                ranges2_1 = idx2ranges(idx2_1);
                ranges2_0 = idx2ranges(idx2_0);

                fprintf('Cluster %d:\n',k);

                % 打印该簇的大小
                n_k = sum(clusterID == k);
                fprintf('  Size: %d solutions\n', n_k);

                % x1 部分
                if ~isempty(ranges1_1)
                    fprintf('  x1 bits strongly 1 at: %s\n', ranges2str(ranges1_1));
                else
                    fprintf('  x1 bits strongly 1 at: (none)\n');
                end
                if ~isempty(ranges1_0)
                    fprintf('  x1 bits strongly 0 at: %s\n', ranges2str(ranges1_0));
                else
                    fprintf('  x1 bits strongly 0 at: (none)\n');
                end

                % x2 部分
                if ~isempty(ranges2_1)
                    fprintf('  x2 bits strongly 1 at: %s\n', ranges2str(ranges2_1));
                else
                    fprintf('  x2 bits strongly 1 at: (none)\n');
                end
                if ~isempty(ranges2_0)
                    fprintf('  x2 bits strongly 0 at: %s\n', ranges2str(ranges2_0));
                else
                    fprintf('  x2 bits strongly 0 at: (none)\n');
                end

                fprintf('\n');
            end
        end

    end
end

%% ---- helper: decode binary string to (x1,x2) in [0,1]^2 ----
function X = decodeBinaryToXY(obj,PopDec)
    % PopDec: N x D binary
    N = size(PopDec,1);
    b  = obj.bitsPerDim;
    D  = 2*b;
    if size(PopDec,2) ~= D
        error('Unexpected PopDec size: expected %d bits, got %d.',D,size(PopDec,2));
    end

    bits1 = PopDec(:,1:b);         % N x b
    bits2 = PopDec(:,b+1:2*b);     % N x b

    % binary to integer (left-msb)
    powers = 2.^(b-1:-1:0);        % 1 x b
    c1 = double(bits1) * powers';  % N x 1
    c2 = double(bits2) * powers';  % N x 1

    % map to [0,1]
    maxVal = (2^b - 1);
    x1 = c1 / maxVal;
    x2 = c2 / maxVal;

    X = [x1, x2];                  % N x 2
end

%% ---- helper: regular polygon vertices (2 x K) ----
function V = regularPolygonVertices(cx,cy,r,K,theta0)
    if nargin < 5
        theta0 = 0;
    end
    theta = theta0 + (0:K-1)*(2*pi/K);
    V = zeros(2,K);
    V(1,:) = cx + r*cos(theta);
    V(2,:) = cy + r*sin(theta);
end

%% ---- helper: compute outward normals and offsets for polygon edges ----
function [N, dvec] = polygonNormals(V)
    % V: 2 x K list of vertices (in order, counter-clockwise)
    K = size(V,2);
    N = zeros(2,K);
    dvec = zeros(1,K);
    for k = 1:K
        i1 = k;
        i2 = mod(k,K) + 1;
        p1 = V(:,i1);
        p2 = V(:,i2);
        e  = p2 - p1;                 % edge direction
        % outward normal (rotate edge by +90 degrees)
        n  = [ e(2); -e(1) ];
        N(:,k)   = n;
        dvec(k)  = dot(n,p1);
    end
end

%% ---- helper: indices -> ranges (e.g., [1 2 3 7 8] -> [1 3; 7 8]) ----
function ranges = idx2ranges(idx)
    idx = idx(:)';
    if isempty(idx)
        ranges = [];
        return;
    end
    idx = sort(idx);
    startIdx = idx(1);
    prev = idx(1);
    ranges = [];
    for t = idx(2:end)
        if t == prev + 1
            prev = t;
        else
            ranges = [ranges; startIdx prev];
            startIdx = t;
            prev = t;
        end
    end
    ranges = [ranges; startIdx prev];
end

%% ---- helper: turn ranges into a compact string ----
function s = ranges2str(ranges)
    % ranges: m x 2
    parts = cell(1,size(ranges,1));
    for i = 1:size(ranges,1)
        a = ranges(i,1);
        b = ranges(i,2);
        if a == b
            parts{i} = sprintf('%d',a);
        else
            parts{i} = sprintf('%d-%d',a,b);
        end
    end
    s = strjoin(parts, ', ');
end
