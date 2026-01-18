import math
from functools import partial
import pytest
import networkx as nx
class TestResourceAllocationIndex:

    @classmethod
    def setup_class(cls):
        cls.func = staticmethod(nx.resource_allocation_index)
        cls.test = partial(_test_func, predict_func=cls.func)

    def test_K5(self):
        G = nx.complete_graph(5)
        self.test(G, [(0, 1)], [(0, 1, 0.75)])

    def test_P3(self):
        G = nx.path_graph(3)
        self.test(G, [(0, 2)], [(0, 2, 0.5)])

    def test_S4(self):
        G = nx.star_graph(4)
        self.test(G, [(1, 2)], [(1, 2, 0.25)])

    def test_notimplemented(self):
        assert pytest.raises(nx.NetworkXNotImplemented, self.func, nx.DiGraph([(0, 1), (1, 2)]), [(0, 2)])
        assert pytest.raises(nx.NetworkXNotImplemented, self.func, nx.MultiGraph([(0, 1), (1, 2)]), [(0, 2)])
        assert pytest.raises(nx.NetworkXNotImplemented, self.func, nx.MultiDiGraph([(0, 1), (1, 2)]), [(0, 2)])

    def test_no_common_neighbor(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        self.test(G, [(0, 1)], [(0, 1, 0)])

    def test_equal_nodes(self):
        G = nx.complete_graph(4)
        self.test(G, [(0, 0)], [(0, 0, 1)])

    def test_all_nonexistent_edges(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (2, 3)])
        self.test(G, None, [(0, 3, 0.5), (1, 2, 0.5), (1, 3, 0)])