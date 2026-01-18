import networkx as nx
class TestDepthLimitedSearch:

    @classmethod
    def setup_class(cls):
        G = nx.Graph()
        nx.add_path(G, [0, 1, 2, 3, 4, 5, 6])
        nx.add_path(G, [2, 7, 8, 9, 10])
        cls.G = G
        D = nx.Graph()
        D.add_edges_from([(0, 1), (2, 3)])
        nx.add_path(D, [2, 7, 8, 9, 10])
        cls.D = D

    def test_dls_preorder_nodes(self):
        assert list(nx.dfs_preorder_nodes(self.G, source=0, depth_limit=2)) == [0, 1, 2]
        assert list(nx.dfs_preorder_nodes(self.D, source=1, depth_limit=2)) == [1, 0]

    def test_dls_postorder_nodes(self):
        assert list(nx.dfs_postorder_nodes(self.G, source=3, depth_limit=3)) == [1, 7, 2, 5, 4, 3]
        assert list(nx.dfs_postorder_nodes(self.D, source=2, depth_limit=2)) == [3, 7, 2]

    def test_dls_successor(self):
        result = nx.dfs_successors(self.G, source=4, depth_limit=3)
        assert {n: set(v) for n, v in result.items()} == {2: {1, 7}, 3: {2}, 4: {3, 5}, 5: {6}}
        result = nx.dfs_successors(self.D, source=7, depth_limit=2)
        assert {n: set(v) for n, v in result.items()} == {8: {9}, 2: {3}, 7: {8, 2}}

    def test_dls_predecessor(self):
        assert nx.dfs_predecessors(self.G, source=0, depth_limit=3) == {1: 0, 2: 1, 3: 2, 7: 2}
        assert nx.dfs_predecessors(self.D, source=2, depth_limit=3) == {8: 7, 9: 8, 3: 2, 7: 2}

    def test_dls_tree(self):
        T = nx.dfs_tree(self.G, source=3, depth_limit=1)
        assert sorted(T.edges()) == [(3, 2), (3, 4)]

    def test_dls_edges(self):
        edges = nx.dfs_edges(self.G, source=9, depth_limit=4)
        assert list(edges) == [(9, 8), (8, 7), (7, 2), (2, 1), (2, 3), (9, 10)]

    def test_dls_labeled_edges_depth_1(self):
        edges = list(nx.dfs_labeled_edges(self.G, source=5, depth_limit=1))
        forward = [(u, v) for u, v, d in edges if d == 'forward']
        assert forward == [(5, 5), (5, 4), (5, 6)]
        assert edges == [(5, 5, 'forward'), (5, 4, 'forward'), (5, 4, 'reverse-depth_limit'), (5, 6, 'forward'), (5, 6, 'reverse-depth_limit'), (5, 5, 'reverse')]

    def test_dls_labeled_edges_depth_2(self):
        edges = list(nx.dfs_labeled_edges(self.G, source=6, depth_limit=2))
        forward = [(u, v) for u, v, d in edges if d == 'forward']
        assert forward == [(6, 6), (6, 5), (5, 4)]
        assert edges == [(6, 6, 'forward'), (6, 5, 'forward'), (5, 4, 'forward'), (5, 4, 'reverse-depth_limit'), (5, 6, 'nontree'), (6, 5, 'reverse'), (6, 6, 'reverse')]

    def test_dls_labeled_disconnected_edges(self):
        edges = list(nx.dfs_labeled_edges(self.D, depth_limit=1))
        assert edges == [(0, 0, 'forward'), (0, 1, 'forward'), (0, 1, 'reverse-depth_limit'), (0, 0, 'reverse'), (2, 2, 'forward'), (2, 3, 'forward'), (2, 3, 'reverse-depth_limit'), (2, 7, 'forward'), (2, 7, 'reverse-depth_limit'), (2, 2, 'reverse'), (8, 8, 'forward'), (8, 7, 'nontree'), (8, 9, 'forward'), (8, 9, 'reverse-depth_limit'), (8, 8, 'reverse'), (10, 10, 'forward'), (10, 9, 'nontree'), (10, 10, 'reverse')]
        edges = list(nx.dfs_labeled_edges(self.D, depth_limit=19))
        assert edges == [(0, 0, 'forward'), (0, 1, 'forward'), (1, 0, 'nontree'), (0, 1, 'reverse'), (0, 0, 'reverse'), (2, 2, 'forward'), (2, 3, 'forward'), (3, 2, 'nontree'), (2, 3, 'reverse'), (2, 7, 'forward'), (7, 2, 'nontree'), (7, 8, 'forward'), (8, 7, 'nontree'), (8, 9, 'forward'), (9, 8, 'nontree'), (9, 10, 'forward'), (10, 9, 'nontree'), (9, 10, 'reverse'), (8, 9, 'reverse'), (7, 8, 'reverse'), (2, 7, 'reverse'), (2, 2, 'reverse')]