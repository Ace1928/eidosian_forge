import networkx as nx
class TestDFS:

    @classmethod
    def setup_class(cls):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 4), (3, 0), (0, 4)])
        cls.G = G
        D = nx.Graph()
        D.add_edges_from([(0, 1), (2, 3)])
        cls.D = D

    def test_preorder_nodes(self):
        assert list(nx.dfs_preorder_nodes(self.G, source=0)) == [0, 1, 2, 4, 3]
        assert list(nx.dfs_preorder_nodes(self.D)) == [0, 1, 2, 3]
        assert list(nx.dfs_preorder_nodes(self.D, source=2)) == [2, 3]

    def test_postorder_nodes(self):
        assert list(nx.dfs_postorder_nodes(self.G, source=0)) == [4, 2, 3, 1, 0]
        assert list(nx.dfs_postorder_nodes(self.D)) == [1, 0, 3, 2]
        assert list(nx.dfs_postorder_nodes(self.D, source=0)) == [1, 0]

    def test_successor(self):
        assert nx.dfs_successors(self.G, source=0) == {0: [1], 1: [2, 3], 2: [4]}
        assert nx.dfs_successors(self.G, source=1) == {0: [3, 4], 1: [0], 4: [2]}
        assert nx.dfs_successors(self.D) == {0: [1], 2: [3]}
        assert nx.dfs_successors(self.D, source=1) == {1: [0]}

    def test_predecessor(self):
        assert nx.dfs_predecessors(self.G, source=0) == {1: 0, 2: 1, 3: 1, 4: 2}
        assert nx.dfs_predecessors(self.D) == {1: 0, 3: 2}

    def test_dfs_tree(self):
        exp_nodes = sorted(self.G.nodes())
        exp_edges = [(0, 1), (1, 2), (1, 3), (2, 4)]
        T = nx.dfs_tree(self.G, source=0)
        assert sorted(T.nodes()) == exp_nodes
        assert sorted(T.edges()) == exp_edges
        T = nx.dfs_tree(self.G, source=None)
        assert sorted(T.nodes()) == exp_nodes
        assert sorted(T.edges()) == exp_edges
        T = nx.dfs_tree(self.G)
        assert sorted(T.nodes()) == exp_nodes
        assert sorted(T.edges()) == exp_edges

    def test_dfs_edges(self):
        edges = nx.dfs_edges(self.G, source=0)
        assert list(edges) == [(0, 1), (1, 2), (2, 4), (1, 3)]
        edges = nx.dfs_edges(self.D)
        assert list(edges) == [(0, 1), (2, 3)]

    def test_dfs_labeled_edges(self):
        edges = list(nx.dfs_labeled_edges(self.G, source=0))
        forward = [(u, v) for u, v, d in edges if d == 'forward']
        assert forward == [(0, 0), (0, 1), (1, 2), (2, 4), (1, 3)]
        assert edges == [(0, 0, 'forward'), (0, 1, 'forward'), (1, 0, 'nontree'), (1, 2, 'forward'), (2, 1, 'nontree'), (2, 4, 'forward'), (4, 2, 'nontree'), (4, 0, 'nontree'), (2, 4, 'reverse'), (1, 2, 'reverse'), (1, 3, 'forward'), (3, 1, 'nontree'), (3, 0, 'nontree'), (1, 3, 'reverse'), (0, 1, 'reverse'), (0, 3, 'nontree'), (0, 4, 'nontree'), (0, 0, 'reverse')]

    def test_dfs_labeled_disconnected_edges(self):
        edges = list(nx.dfs_labeled_edges(self.D))
        forward = [(u, v) for u, v, d in edges if d == 'forward']
        assert forward == [(0, 0), (0, 1), (2, 2), (2, 3)]
        assert edges == [(0, 0, 'forward'), (0, 1, 'forward'), (1, 0, 'nontree'), (0, 1, 'reverse'), (0, 0, 'reverse'), (2, 2, 'forward'), (2, 3, 'forward'), (3, 2, 'nontree'), (2, 3, 'reverse'), (2, 2, 'reverse')]

    def test_dfs_tree_isolates(self):
        G = nx.Graph()
        G.add_node(1)
        G.add_node(2)
        T = nx.dfs_tree(G, source=1)
        assert sorted(T.nodes()) == [1]
        assert sorted(T.edges()) == []
        T = nx.dfs_tree(G, source=None)
        assert sorted(T.nodes()) == [1, 2]
        assert sorted(T.edges()) == []