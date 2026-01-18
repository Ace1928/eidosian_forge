import pytest
import networkx as nx
from networkx.utils import nodes_equal
from .test_graph import BaseAttrGraphTester, BaseGraphTester
from .test_graph import TestEdgeSubgraph as _TestGraphEdgeSubgraph
from .test_graph import TestGraph as _TestGraph
class TestDiGraph(BaseAttrDiGraphTester, _TestGraph):
    """Tests specific to dict-of-dict-of-dict digraph data structure"""

    def setup_method(self):
        self.Graph = nx.DiGraph
        ed1, ed2, ed3, ed4, ed5, ed6 = ({}, {}, {}, {}, {}, {})
        self.k3adj = {0: {1: ed1, 2: ed2}, 1: {0: ed3, 2: ed4}, 2: {0: ed5, 1: ed6}}
        self.k3edges = [(0, 1), (0, 2), (1, 2)]
        self.k3nodes = [0, 1, 2]
        self.K3 = self.Graph()
        self.K3._succ = self.k3adj
        self.K3._pred = {0: {1: ed3, 2: ed5}, 1: {0: ed1, 2: ed6}, 2: {0: ed2, 1: ed4}}
        self.K3._node = {}
        self.K3._node[0] = {}
        self.K3._node[1] = {}
        self.K3._node[2] = {}
        ed1, ed2 = ({}, {})
        self.P3 = self.Graph()
        self.P3._succ = {0: {1: ed1}, 1: {2: ed2}, 2: {}}
        self.P3._pred = {0: {}, 1: {0: ed1}, 2: {1: ed2}}
        self.P3._node = {}
        self.P3._node[0] = {}
        self.P3._node[1] = {}
        self.P3._node[2] = {}

    def test_data_input(self):
        G = self.Graph({1: [2], 2: [1]}, name='test')
        assert G.name == 'test'
        assert sorted(G.adj.items()) == [(1, {2: {}}), (2, {1: {}})]
        assert sorted(G.succ.items()) == [(1, {2: {}}), (2, {1: {}})]
        assert sorted(G.pred.items()) == [(1, {2: {}}), (2, {1: {}})]

    def test_add_edge(self):
        G = self.Graph()
        G.add_edge(0, 1)
        assert G.adj == {0: {1: {}}, 1: {}}
        assert G.succ == {0: {1: {}}, 1: {}}
        assert G.pred == {0: {}, 1: {0: {}}}
        G = self.Graph()
        G.add_edge(*(0, 1))
        assert G.adj == {0: {1: {}}, 1: {}}
        assert G.succ == {0: {1: {}}, 1: {}}
        assert G.pred == {0: {}, 1: {0: {}}}
        with pytest.raises(ValueError, match='None cannot be a node'):
            G.add_edge(None, 3)

    def test_add_edges_from(self):
        G = self.Graph()
        G.add_edges_from([(0, 1), (0, 2, {'data': 3})], data=2)
        assert G.adj == {0: {1: {'data': 2}, 2: {'data': 3}}, 1: {}, 2: {}}
        assert G.succ == {0: {1: {'data': 2}, 2: {'data': 3}}, 1: {}, 2: {}}
        assert G.pred == {0: {}, 1: {0: {'data': 2}}, 2: {0: {'data': 3}}}
        with pytest.raises(nx.NetworkXError):
            G.add_edges_from([(0,)])
        with pytest.raises(nx.NetworkXError):
            G.add_edges_from([(0, 1, 2, 3)])
        with pytest.raises(TypeError):
            G.add_edges_from([0])
        with pytest.raises(ValueError, match='None cannot be a node'):
            G.add_edges_from([(None, 3), (3, 2)])

    def test_remove_edge(self):
        G = self.K3.copy()
        G.remove_edge(0, 1)
        assert G.succ == {0: {2: {}}, 1: {0: {}, 2: {}}, 2: {0: {}, 1: {}}}
        assert G.pred == {0: {1: {}, 2: {}}, 1: {2: {}}, 2: {0: {}, 1: {}}}
        with pytest.raises(nx.NetworkXError):
            G.remove_edge(-1, 0)

    def test_remove_edges_from(self):
        G = self.K3.copy()
        G.remove_edges_from([(0, 1)])
        assert G.succ == {0: {2: {}}, 1: {0: {}, 2: {}}, 2: {0: {}, 1: {}}}
        assert G.pred == {0: {1: {}, 2: {}}, 1: {2: {}}, 2: {0: {}, 1: {}}}
        G.remove_edges_from([(0, 0)])

    def test_clear(self):
        G = self.K3
        G.graph['name'] = 'K3'
        G.clear()
        assert list(G.nodes) == []
        assert G.succ == {}
        assert G.pred == {}
        assert G.graph == {}

    def test_clear_edges(self):
        G = self.K3
        G.graph['name'] = 'K3'
        nodes = list(G.nodes)
        G.clear_edges()
        assert list(G.nodes) == nodes
        expected = {0: {}, 1: {}, 2: {}}
        assert G.succ == expected
        assert G.pred == expected
        assert list(G.edges) == []
        assert G.graph['name'] == 'K3'