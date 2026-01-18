import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
class TestNodeViewSetOps:

    @classmethod
    def setup_class(cls):
        cls.G = nx.path_graph(9)
        cls.G.nodes[3]['foo'] = 'bar'
        cls.nv = cls.G.nodes

    def n_its(self, nodes):
        return set(nodes)

    def test_len(self):
        G = self.G.copy()
        nv = G.nodes
        assert len(nv) == 9
        G.remove_node(7)
        assert len(nv) == 8
        G.add_node(9)
        assert len(nv) == 9

    def test_and(self):
        nv = self.nv
        some_nodes = self.n_its(range(5, 12))
        assert nv & some_nodes == self.n_its(range(5, 9))
        assert some_nodes & nv == self.n_its(range(5, 9))

    def test_or(self):
        nv = self.nv
        some_nodes = self.n_its(range(5, 12))
        assert nv | some_nodes == self.n_its(range(12))
        assert some_nodes | nv == self.n_its(range(12))

    def test_xor(self):
        nv = self.nv
        some_nodes = self.n_its(range(5, 12))
        nodes = {0, 1, 2, 3, 4, 9, 10, 11}
        assert nv ^ some_nodes == self.n_its(nodes)
        assert some_nodes ^ nv == self.n_its(nodes)

    def test_sub(self):
        nv = self.nv
        some_nodes = self.n_its(range(5, 12))
        assert nv - some_nodes == self.n_its(range(5))
        assert some_nodes - nv == self.n_its(range(9, 12))