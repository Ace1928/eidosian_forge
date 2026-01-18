from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_multigraph import BaseMultiGraphTester
from .test_multigraph import TestEdgeSubgraph as _TestMultiGraphEdgeSubgraph
from .test_multigraph import TestMultiGraph as _TestMultiGraph
class TestMultiDiGraph(BaseMultiDiGraphTester, _TestMultiGraph):

    def setup_method(self):
        self.Graph = nx.MultiDiGraph
        self.k3edges = [(0, 1), (0, 2), (1, 2)]
        self.k3nodes = [0, 1, 2]
        self.K3 = self.Graph()
        self.K3._succ = {0: {}, 1: {}, 2: {}}
        self.K3._pred = {0: {}, 1: {}, 2: {}}
        for u in self.k3nodes:
            for v in self.k3nodes:
                if u == v:
                    continue
                d = {0: {}}
                self.K3._succ[u][v] = d
                self.K3._pred[v][u] = d
        self.K3._node = {}
        self.K3._node[0] = {}
        self.K3._node[1] = {}
        self.K3._node[2] = {}

    def test_add_edge(self):
        G = self.Graph()
        G.add_edge(0, 1)
        assert G._adj == {0: {1: {0: {}}}, 1: {}}
        assert G._succ == {0: {1: {0: {}}}, 1: {}}
        assert G._pred == {0: {}, 1: {0: {0: {}}}}
        G = self.Graph()
        G.add_edge(*(0, 1))
        assert G._adj == {0: {1: {0: {}}}, 1: {}}
        assert G._succ == {0: {1: {0: {}}}, 1: {}}
        assert G._pred == {0: {}, 1: {0: {0: {}}}}
        with pytest.raises(ValueError, match='None cannot be a node'):
            G.add_edge(None, 3)

    def test_add_edges_from(self):
        G = self.Graph()
        G.add_edges_from([(0, 1), (0, 1, {'weight': 3})])
        assert G._adj == {0: {1: {0: {}, 1: {'weight': 3}}}, 1: {}}
        assert G._succ == {0: {1: {0: {}, 1: {'weight': 3}}}, 1: {}}
        assert G._pred == {0: {}, 1: {0: {0: {}, 1: {'weight': 3}}}}
        G.add_edges_from([(0, 1), (0, 1, {'weight': 3})], weight=2)
        assert G._succ == {0: {1: {0: {}, 1: {'weight': 3}, 2: {'weight': 2}, 3: {'weight': 3}}}, 1: {}}
        assert G._pred == {0: {}, 1: {0: {0: {}, 1: {'weight': 3}, 2: {'weight': 2}, 3: {'weight': 3}}}}
        G = self.Graph()
        edges = [(0, 1, {'weight': 3}), (0, 1, (('weight', 2),)), (0, 1, 5), (0, 1, 's')]
        G.add_edges_from(edges)
        keydict = {0: {'weight': 3}, 1: {'weight': 2}, 5: {}, 's': {}}
        assert G._succ == {0: {1: keydict}, 1: {}}
        assert G._pred == {1: {0: keydict}, 0: {}}
        pytest.raises(nx.NetworkXError, G.add_edges_from, [(0,)])
        pytest.raises(nx.NetworkXError, G.add_edges_from, [(0, 1, 2, 3, 4)])
        pytest.raises(TypeError, G.add_edges_from, [0])
        with pytest.raises(ValueError, match='None cannot be a node'):
            G.add_edges_from([(None, 3), (3, 2)])

    def test_remove_edge(self):
        G = self.K3
        G.remove_edge(0, 1)
        assert G._succ == {0: {2: {0: {}}}, 1: {0: {0: {}}, 2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        assert G._pred == {0: {1: {0: {}}, 2: {0: {}}}, 1: {2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        pytest.raises((KeyError, nx.NetworkXError), G.remove_edge, -1, 0)
        pytest.raises((KeyError, nx.NetworkXError), G.remove_edge, 0, 2, key=1)

    def test_remove_multiedge(self):
        G = self.K3
        G.add_edge(0, 1, key='parallel edge')
        G.remove_edge(0, 1, key='parallel edge')
        assert G._adj == {0: {1: {0: {}}, 2: {0: {}}}, 1: {0: {0: {}}, 2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        assert G._succ == {0: {1: {0: {}}, 2: {0: {}}}, 1: {0: {0: {}}, 2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        assert G._pred == {0: {1: {0: {}}, 2: {0: {}}}, 1: {0: {0: {}}, 2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        G.remove_edge(0, 1)
        assert G._succ == {0: {2: {0: {}}}, 1: {0: {0: {}}, 2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        assert G._pred == {0: {1: {0: {}}, 2: {0: {}}}, 1: {2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        pytest.raises((KeyError, nx.NetworkXError), G.remove_edge, -1, 0)

    def test_remove_edges_from(self):
        G = self.K3
        G.remove_edges_from([(0, 1)])
        assert G._succ == {0: {2: {0: {}}}, 1: {0: {0: {}}, 2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        assert G._pred == {0: {1: {0: {}}, 2: {0: {}}}, 1: {2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        G.remove_edges_from([(0, 0)])