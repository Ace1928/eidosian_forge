import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
class BaseGraphTester:
    """Tests for data-structure independent graph class features."""

    def test_contains(self):
        G = self.K3
        assert 1 in G
        assert 4 not in G
        assert 'b' not in G
        assert [] not in G
        assert {1: 1} not in G

    def test_order(self):
        G = self.K3
        assert len(G) == 3
        assert G.order() == 3
        assert G.number_of_nodes() == 3

    def test_nodes(self):
        G = self.K3
        assert isinstance(G._node, G.node_dict_factory)
        assert isinstance(G._adj, G.adjlist_outer_dict_factory)
        assert all((isinstance(adj, G.adjlist_inner_dict_factory) for adj in G._adj.values()))
        assert sorted(G.nodes()) == self.k3nodes
        assert sorted(G.nodes(data=True)) == [(0, {}), (1, {}), (2, {})]

    def test_none_node(self):
        G = self.Graph()
        with pytest.raises(ValueError):
            G.add_node(None)
        with pytest.raises(ValueError):
            G.add_nodes_from([None])
        with pytest.raises(ValueError):
            G.add_edge(0, None)
        with pytest.raises(ValueError):
            G.add_edges_from([(0, None)])

    def test_has_node(self):
        G = self.K3
        assert G.has_node(1)
        assert not G.has_node(4)
        assert not G.has_node([])
        assert not G.has_node({1: 1})

    def test_has_edge(self):
        G = self.K3
        assert G.has_edge(0, 1)
        assert not G.has_edge(0, -1)

    def test_neighbors(self):
        G = self.K3
        assert sorted(G.neighbors(0)) == [1, 2]
        with pytest.raises(nx.NetworkXError):
            G.neighbors(-1)

    @pytest.mark.skipif(platform.python_implementation() == 'PyPy', reason='PyPy gc is different')
    def test_memory_leak(self):
        G = self.Graph()

        def count_objects_of_type(_type):
            return sum((1 for obj in gc.get_objects() if not isinstance(obj, weakref.ProxyTypes) and isinstance(obj, _type)))
        gc.collect()
        before = count_objects_of_type(self.Graph)
        G.copy()
        gc.collect()
        after = count_objects_of_type(self.Graph)
        assert before == after

        class MyGraph(self.Graph):
            pass
        gc.collect()
        G = MyGraph()
        before = count_objects_of_type(MyGraph)
        G.copy()
        gc.collect()
        after = count_objects_of_type(MyGraph)
        assert before == after

    def test_edges(self):
        G = self.K3
        assert isinstance(G._adj, G.adjlist_outer_dict_factory)
        assert edges_equal(G.edges(), [(0, 1), (0, 2), (1, 2)])
        assert edges_equal(G.edges(0), [(0, 1), (0, 2)])
        assert edges_equal(G.edges([0, 1]), [(0, 1), (0, 2), (1, 2)])
        with pytest.raises(nx.NetworkXError):
            G.edges(-1)

    def test_degree(self):
        G = self.K3
        assert sorted(G.degree()) == [(0, 2), (1, 2), (2, 2)]
        assert dict(G.degree()) == {0: 2, 1: 2, 2: 2}
        assert G.degree(0) == 2
        with pytest.raises(nx.NetworkXError):
            G.degree(-1)

    def test_size(self):
        G = self.K3
        assert G.size() == 3
        assert G.number_of_edges() == 3

    def test_nbunch_iter(self):
        G = self.K3
        assert nodes_equal(G.nbunch_iter(), self.k3nodes)
        assert nodes_equal(G.nbunch_iter(0), [0])
        assert nodes_equal(G.nbunch_iter([0, 1]), [0, 1])
        assert nodes_equal(G.nbunch_iter([-1]), [])
        assert nodes_equal(G.nbunch_iter('foo'), [])
        bunch = G.nbunch_iter(-1)
        with pytest.raises(nx.NetworkXError, match='is not a node or a sequence'):
            list(bunch)
        bunch = G.nbunch_iter([0, 1, 2, {}])
        with pytest.raises(nx.NetworkXError, match='in sequence nbunch is not a valid node'):
            list(bunch)

    def test_nbunch_iter_node_format_raise(self):
        G = self.Graph()
        nbunch = [('x', set())]
        with pytest.raises(nx.NetworkXError):
            list(G.nbunch_iter(nbunch))

    def test_selfloop_degree(self):
        G = self.Graph()
        G.add_edge(1, 1)
        assert sorted(G.degree()) == [(1, 2)]
        assert dict(G.degree()) == {1: 2}
        assert G.degree(1) == 2
        assert sorted(G.degree([1])) == [(1, 2)]
        assert G.degree(1, weight='weight') == 2

    def test_selfloops(self):
        G = self.K3.copy()
        G.add_edge(0, 0)
        assert nodes_equal(nx.nodes_with_selfloops(G), [0])
        assert edges_equal(nx.selfloop_edges(G), [(0, 0)])
        assert nx.number_of_selfloops(G) == 1
        G.remove_edge(0, 0)
        G.add_edge(0, 0)
        G.remove_edges_from([(0, 0)])
        G.add_edge(1, 1)
        G.remove_node(1)
        G.add_edge(0, 0)
        G.add_edge(1, 1)
        G.remove_nodes_from([0, 1])

    def test_cache_reset(self):
        G = self.K3.copy()
        old_adj = G.adj
        assert id(G.adj) == id(old_adj)
        G._adj = {}
        assert id(G.adj) != id(old_adj)
        old_nodes = G.nodes
        assert id(G.nodes) == id(old_nodes)
        G._node = {}
        assert id(G.nodes) != id(old_nodes)

    def test_attributes_cached(self):
        G = self.K3.copy()
        assert id(G.nodes) == id(G.nodes)
        assert id(G.edges) == id(G.edges)
        assert id(G.degree) == id(G.degree)
        assert id(G.adj) == id(G.adj)