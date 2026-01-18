from functools import partial
import pytest
import networkx as nx
class TestBreadthLimitedSearch:

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

    def test_limited_bfs_successor(self):
        assert dict(nx.bfs_successors(self.G, source=1, depth_limit=3)) == {1: [0, 2], 2: [3, 7], 3: [4], 7: [8]}
        result = {n: sorted(s) for n, s in nx.bfs_successors(self.D, source=7, depth_limit=2)}
        assert result == {8: [9], 2: [3], 7: [2, 8]}

    def test_limited_bfs_predecessor(self):
        assert dict(nx.bfs_predecessors(self.G, source=1, depth_limit=3)) == {0: 1, 2: 1, 3: 2, 4: 3, 7: 2, 8: 7}
        assert dict(nx.bfs_predecessors(self.D, source=7, depth_limit=2)) == {2: 7, 3: 2, 8: 7, 9: 8}

    def test_limited_bfs_tree(self):
        T = nx.bfs_tree(self.G, source=3, depth_limit=1)
        assert sorted(T.edges()) == [(3, 2), (3, 4)]

    def test_limited_bfs_edges(self):
        edges = nx.bfs_edges(self.G, source=9, depth_limit=4)
        assert list(edges) == [(9, 8), (9, 10), (8, 7), (7, 2), (2, 1), (2, 3)]

    def test_limited_bfs_layers(self):
        assert dict(enumerate(nx.bfs_layers(self.G, sources=[0]))) == {0: [0], 1: [1], 2: [2], 3: [3, 7], 4: [4, 8], 5: [5, 9], 6: [6, 10]}
        assert dict(enumerate(nx.bfs_layers(self.D, sources=2))) == {0: [2], 1: [3, 7], 2: [8], 3: [9], 4: [10]}

    def test_limited_descendants_at_distance(self):
        for distance, descendants in enumerate([{0}, {1}, {2}, {3, 7}, {4, 8}, {5, 9}, {6, 10}]):
            assert nx.descendants_at_distance(self.G, 0, distance) == descendants
        for distance, descendants in enumerate([{2}, {3, 7}, {8}, {9}, {10}]):
            assert nx.descendants_at_distance(self.D, 2, distance) == descendants