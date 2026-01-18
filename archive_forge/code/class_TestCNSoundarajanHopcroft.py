import math
from functools import partial
import pytest
import networkx as nx
class TestCNSoundarajanHopcroft:

    @classmethod
    def setup_class(cls):
        cls.func = staticmethod(nx.cn_soundarajan_hopcroft)
        cls.test = partial(_test_func, predict_func=cls.func, community='community')

    def test_K5(self):
        G = nx.complete_graph(5)
        G.nodes[0]['community'] = 0
        G.nodes[1]['community'] = 0
        G.nodes[2]['community'] = 0
        G.nodes[3]['community'] = 0
        G.nodes[4]['community'] = 1
        self.test(G, [(0, 1)], [(0, 1, 5)])

    def test_P3(self):
        G = nx.path_graph(3)
        G.nodes[0]['community'] = 0
        G.nodes[1]['community'] = 1
        G.nodes[2]['community'] = 0
        self.test(G, [(0, 2)], [(0, 2, 1)])

    def test_S4(self):
        G = nx.star_graph(4)
        G.nodes[0]['community'] = 1
        G.nodes[1]['community'] = 1
        G.nodes[2]['community'] = 1
        G.nodes[3]['community'] = 0
        G.nodes[4]['community'] = 0
        self.test(G, [(1, 2)], [(1, 2, 2)])

    def test_notimplemented(self):
        G = nx.DiGraph([(0, 1), (1, 2)])
        G.add_nodes_from([0, 1, 2], community=0)
        assert pytest.raises(nx.NetworkXNotImplemented, self.func, G, [(0, 2)])
        G = nx.MultiGraph([(0, 1), (1, 2)])
        G.add_nodes_from([0, 1, 2], community=0)
        assert pytest.raises(nx.NetworkXNotImplemented, self.func, G, [(0, 2)])
        G = nx.MultiDiGraph([(0, 1), (1, 2)])
        G.add_nodes_from([0, 1, 2], community=0)
        assert pytest.raises(nx.NetworkXNotImplemented, self.func, G, [(0, 2)])

    def test_no_common_neighbor(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        G.nodes[0]['community'] = 0
        G.nodes[1]['community'] = 0
        self.test(G, [(0, 1)], [(0, 1, 0)])

    def test_equal_nodes(self):
        G = nx.complete_graph(3)
        G.nodes[0]['community'] = 0
        G.nodes[1]['community'] = 0
        G.nodes[2]['community'] = 0
        self.test(G, [(0, 0)], [(0, 0, 4)])

    def test_different_community(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
        G.nodes[0]['community'] = 0
        G.nodes[1]['community'] = 0
        G.nodes[2]['community'] = 0
        G.nodes[3]['community'] = 1
        self.test(G, [(0, 3)], [(0, 3, 2)])

    def test_no_community_information(self):
        G = nx.complete_graph(5)
        assert pytest.raises(nx.NetworkXAlgorithmError, list, self.func(G, [(0, 1)]))

    def test_insufficient_community_information(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
        G.nodes[0]['community'] = 0
        G.nodes[1]['community'] = 0
        G.nodes[3]['community'] = 0
        assert pytest.raises(nx.NetworkXAlgorithmError, list, self.func(G, [(0, 3)]))

    def test_sufficient_community_information(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])
        G.nodes[1]['community'] = 0
        G.nodes[2]['community'] = 0
        G.nodes[3]['community'] = 0
        G.nodes[4]['community'] = 0
        self.test(G, [(1, 4)], [(1, 4, 4)])

    def test_custom_community_attribute_name(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
        G.nodes[0]['cmty'] = 0
        G.nodes[1]['cmty'] = 0
        G.nodes[2]['cmty'] = 0
        G.nodes[3]['cmty'] = 1
        self.test(G, [(0, 3)], [(0, 3, 2)], community='cmty')

    def test_all_nonexistent_edges(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (2, 3)])
        G.nodes[0]['community'] = 0
        G.nodes[1]['community'] = 1
        G.nodes[2]['community'] = 0
        G.nodes[3]['community'] = 0
        self.test(G, None, [(0, 3, 2), (1, 2, 1), (1, 3, 0)])