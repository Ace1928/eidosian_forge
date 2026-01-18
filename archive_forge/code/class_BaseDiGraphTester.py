import pytest
import networkx as nx
from networkx.utils import nodes_equal
from .test_graph import BaseAttrGraphTester, BaseGraphTester
from .test_graph import TestEdgeSubgraph as _TestGraphEdgeSubgraph
from .test_graph import TestGraph as _TestGraph
class BaseDiGraphTester(BaseGraphTester):

    def test_has_successor(self):
        G = self.K3
        assert G.has_successor(0, 1)
        assert not G.has_successor(0, -1)

    def test_successors(self):
        G = self.K3
        assert sorted(G.successors(0)) == [1, 2]
        with pytest.raises(nx.NetworkXError):
            G.successors(-1)

    def test_has_predecessor(self):
        G = self.K3
        assert G.has_predecessor(0, 1)
        assert not G.has_predecessor(0, -1)

    def test_predecessors(self):
        G = self.K3
        assert sorted(G.predecessors(0)) == [1, 2]
        with pytest.raises(nx.NetworkXError):
            G.predecessors(-1)

    def test_edges(self):
        G = self.K3
        assert sorted(G.edges()) == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        assert sorted(G.edges(0)) == [(0, 1), (0, 2)]
        assert sorted(G.edges([0, 1])) == [(0, 1), (0, 2), (1, 0), (1, 2)]
        with pytest.raises(nx.NetworkXError):
            G.edges(-1)

    def test_out_edges(self):
        G = self.K3
        assert sorted(G.out_edges()) == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        assert sorted(G.out_edges(0)) == [(0, 1), (0, 2)]
        with pytest.raises(nx.NetworkXError):
            G.out_edges(-1)

    def test_out_edges_dir(self):
        G = self.P3
        assert sorted(G.out_edges()) == [(0, 1), (1, 2)]
        assert sorted(G.out_edges(0)) == [(0, 1)]
        assert sorted(G.out_edges(2)) == []

    def test_out_edges_data(self):
        G = nx.DiGraph([(0, 1, {'data': 0}), (1, 0, {})])
        assert sorted(G.out_edges(data=True)) == [(0, 1, {'data': 0}), (1, 0, {})]
        assert sorted(G.out_edges(0, data=True)) == [(0, 1, {'data': 0})]
        assert sorted(G.out_edges(data='data')) == [(0, 1, 0), (1, 0, None)]
        assert sorted(G.out_edges(0, data='data')) == [(0, 1, 0)]

    def test_in_edges_dir(self):
        G = self.P3
        assert sorted(G.in_edges()) == [(0, 1), (1, 2)]
        assert sorted(G.in_edges(0)) == []
        assert sorted(G.in_edges(2)) == [(1, 2)]

    def test_in_edges_data(self):
        G = nx.DiGraph([(0, 1, {'data': 0}), (1, 0, {})])
        assert sorted(G.in_edges(data=True)) == [(0, 1, {'data': 0}), (1, 0, {})]
        assert sorted(G.in_edges(1, data=True)) == [(0, 1, {'data': 0})]
        assert sorted(G.in_edges(data='data')) == [(0, 1, 0), (1, 0, None)]
        assert sorted(G.in_edges(1, data='data')) == [(0, 1, 0)]

    def test_degree(self):
        G = self.K3
        assert sorted(G.degree()) == [(0, 4), (1, 4), (2, 4)]
        assert dict(G.degree()) == {0: 4, 1: 4, 2: 4}
        assert G.degree(0) == 4
        assert list(G.degree(iter([0]))) == [(0, 4)]

    def test_in_degree(self):
        G = self.K3
        assert sorted(G.in_degree()) == [(0, 2), (1, 2), (2, 2)]
        assert dict(G.in_degree()) == {0: 2, 1: 2, 2: 2}
        assert G.in_degree(0) == 2
        assert list(G.in_degree(iter([0]))) == [(0, 2)]

    def test_out_degree(self):
        G = self.K3
        assert sorted(G.out_degree()) == [(0, 2), (1, 2), (2, 2)]
        assert dict(G.out_degree()) == {0: 2, 1: 2, 2: 2}
        assert G.out_degree(0) == 2
        assert list(G.out_degree(iter([0]))) == [(0, 2)]

    def test_size(self):
        G = self.K3
        assert G.size() == 6
        assert G.number_of_edges() == 6

    def test_to_undirected_reciprocal(self):
        G = self.Graph()
        G.add_edge(1, 2)
        assert G.to_undirected().has_edge(1, 2)
        assert not G.to_undirected(reciprocal=True).has_edge(1, 2)
        G.add_edge(2, 1)
        assert G.to_undirected(reciprocal=True).has_edge(1, 2)

    def test_reverse_copy(self):
        G = nx.DiGraph([(0, 1), (1, 2)])
        R = G.reverse()
        assert sorted(R.edges()) == [(1, 0), (2, 1)]
        R.remove_edge(1, 0)
        assert sorted(R.edges()) == [(2, 1)]
        assert sorted(G.edges()) == [(0, 1), (1, 2)]

    def test_reverse_nocopy(self):
        G = nx.DiGraph([(0, 1), (1, 2)])
        R = G.reverse(copy=False)
        assert sorted(R.edges()) == [(1, 0), (2, 1)]
        with pytest.raises(nx.NetworkXError):
            R.remove_edge(1, 0)

    def test_reverse_hashable(self):

        class Foo:
            pass
        x = Foo()
        y = Foo()
        G = nx.DiGraph()
        G.add_edge(x, y)
        assert nodes_equal(G.nodes(), G.reverse().nodes())
        assert [(y, x)] == list(G.reverse().edges())

    def test_di_cache_reset(self):
        G = self.K3.copy()
        old_succ = G.succ
        assert id(G.succ) == id(old_succ)
        old_adj = G.adj
        assert id(G.adj) == id(old_adj)
        G._succ = {}
        assert id(G.succ) != id(old_succ)
        assert id(G.adj) != id(old_adj)
        old_pred = G.pred
        assert id(G.pred) == id(old_pred)
        G._pred = {}
        assert id(G.pred) != id(old_pred)

    def test_di_attributes_cached(self):
        G = self.K3.copy()
        assert id(G.in_edges) == id(G.in_edges)
        assert id(G.out_edges) == id(G.out_edges)
        assert id(G.in_degree) == id(G.in_degree)
        assert id(G.out_degree) == id(G.out_degree)
        assert id(G.succ) == id(G.succ)
        assert id(G.pred) == id(G.pred)