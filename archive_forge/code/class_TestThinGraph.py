import networkx as nx
from .test_digraph import BaseDiGraphTester
from .test_digraph import TestDiGraph as _TestDiGraph
from .test_graph import BaseGraphTester
from .test_graph import TestGraph as _TestGraph
from .test_multidigraph import TestMultiDiGraph as _TestMultiDiGraph
from .test_multigraph import TestMultiGraph as _TestMultiGraph
class TestThinGraph(BaseGraphTester):

    def setup_method(self):
        all_edge_dict = {'weight': 1}

        class MyGraph(nx.Graph):

            def edge_attr_dict_factory(self):
                return all_edge_dict
        self.Graph = MyGraph
        ed1, ed2, ed3 = (all_edge_dict, all_edge_dict, all_edge_dict)
        self.k3adj = {0: {1: ed1, 2: ed2}, 1: {0: ed1, 2: ed3}, 2: {0: ed2, 1: ed3}}
        self.k3edges = [(0, 1), (0, 2), (1, 2)]
        self.k3nodes = [0, 1, 2]
        self.K3 = self.Graph()
        self.K3._adj = self.k3adj
        self.K3._node = {}
        self.K3._node[0] = {}
        self.K3._node[1] = {}
        self.K3._node[2] = {}