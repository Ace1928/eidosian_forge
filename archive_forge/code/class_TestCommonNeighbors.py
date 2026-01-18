import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
class TestCommonNeighbors:

    @classmethod
    def setup_class(cls):
        cls.func = staticmethod(nx.common_neighbors)

        def test_func(G, u, v, expected):
            result = sorted(cls.func(G, u, v))
            assert result == expected
        cls.test = staticmethod(test_func)

    def test_K5(self):
        G = nx.complete_graph(5)
        self.test(G, 0, 1, [2, 3, 4])

    def test_P3(self):
        G = nx.path_graph(3)
        self.test(G, 0, 2, [1])

    def test_S4(self):
        G = nx.star_graph(4)
        self.test(G, 1, 2, [0])

    def test_digraph(self):
        with pytest.raises(nx.NetworkXNotImplemented):
            G = nx.DiGraph()
            G.add_edges_from([(0, 1), (1, 2)])
            self.func(G, 0, 2)

    def test_nonexistent_nodes(self):
        G = nx.complete_graph(5)
        pytest.raises(nx.NetworkXError, nx.common_neighbors, G, 5, 4)
        pytest.raises(nx.NetworkXError, nx.common_neighbors, G, 4, 5)
        pytest.raises(nx.NetworkXError, nx.common_neighbors, G, 5, 6)

    def test_custom1(self):
        """Case of no common neighbors."""
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        self.test(G, 0, 1, [])

    def test_custom2(self):
        """Case of equal nodes."""
        G = nx.complete_graph(4)
        self.test(G, 0, 0, [1, 2, 3])