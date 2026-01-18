from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_graph import BaseAttrGraphTester
from .test_graph import TestGraph as _TestGraph
class TestMultiGraphSubclass(TestMultiGraph):

    def setup_method(self):
        self.Graph = MultiGraphSubClass
        self.k3edges = [(0, 1), (0, 2), (1, 2)]
        self.k3nodes = [0, 1, 2]
        self.K3 = self.Graph()
        self.K3._adj = self.K3.adjlist_outer_dict_factory({0: self.K3.adjlist_inner_dict_factory(), 1: self.K3.adjlist_inner_dict_factory(), 2: self.K3.adjlist_inner_dict_factory()})
        self.K3._pred = {0: {}, 1: {}, 2: {}}
        for u in self.k3nodes:
            for v in self.k3nodes:
                if u != v:
                    d = {0: {}}
                    self.K3._adj[u][v] = d
                    self.K3._adj[v][u] = d
        self.K3._node = self.K3.node_dict_factory()
        self.K3._node[0] = self.K3.node_attr_dict_factory()
        self.K3._node[1] = self.K3.node_attr_dict_factory()
        self.K3._node[2] = self.K3.node_attr_dict_factory()