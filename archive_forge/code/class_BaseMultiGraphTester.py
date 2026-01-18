from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_graph import BaseAttrGraphTester
from .test_graph import TestGraph as _TestGraph
class BaseMultiGraphTester(BaseAttrGraphTester):

    def test_has_edge(self):
        G = self.K3
        assert G.has_edge(0, 1)
        assert not G.has_edge(0, -1)
        assert G.has_edge(0, 1, 0)
        assert not G.has_edge(0, 1, 1)

    def test_get_edge_data(self):
        G = self.K3
        assert G.get_edge_data(0, 1) == {0: {}}
        assert G[0][1] == {0: {}}
        assert G[0][1][0] == {}
        assert G.get_edge_data(10, 20) is None
        assert G.get_edge_data(0, 1, 0) == {}

    def test_adjacency(self):
        G = self.K3
        assert dict(G.adjacency()) == {0: {1: {0: {}}, 2: {0: {}}}, 1: {0: {0: {}}, 2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}

    def deepcopy_edge_attr(self, H, G):
        assert G[1][2][0]['foo'] == H[1][2][0]['foo']
        G[1][2][0]['foo'].append(1)
        assert G[1][2][0]['foo'] != H[1][2][0]['foo']

    def shallow_copy_edge_attr(self, H, G):
        assert G[1][2][0]['foo'] == H[1][2][0]['foo']
        G[1][2][0]['foo'].append(1)
        assert G[1][2][0]['foo'] == H[1][2][0]['foo']

    def graphs_equal(self, H, G):
        assert G._adj == H._adj
        assert G._node == H._node
        assert G.graph == H.graph
        assert G.name == H.name
        if not G.is_directed() and (not H.is_directed()):
            assert H._adj[1][2][0] is H._adj[2][1][0]
            assert G._adj[1][2][0] is G._adj[2][1][0]
        else:
            if not G.is_directed():
                G._pred = G._adj
                G._succ = G._adj
            if not H.is_directed():
                H._pred = H._adj
                H._succ = H._adj
            assert G._pred == H._pred
            assert G._succ == H._succ
            assert H._succ[1][2][0] is H._pred[2][1][0]
            assert G._succ[1][2][0] is G._pred[2][1][0]

    def same_attrdict(self, H, G):
        old_foo = H[1][2][0]['foo']
        H.adj[1][2][0]['foo'] = 'baz'
        assert G._adj == H._adj
        H.adj[1][2][0]['foo'] = old_foo
        assert G._adj == H._adj
        old_foo = H.nodes[0]['foo']
        H.nodes[0]['foo'] = 'baz'
        assert G._node == H._node
        H.nodes[0]['foo'] = old_foo
        assert G._node == H._node

    def different_attrdict(self, H, G):
        old_foo = H[1][2][0]['foo']
        H.adj[1][2][0]['foo'] = 'baz'
        assert G._adj != H._adj
        H.adj[1][2][0]['foo'] = old_foo
        assert G._adj == H._adj
        old_foo = H.nodes[0]['foo']
        H.nodes[0]['foo'] = 'baz'
        assert G._node != H._node
        H.nodes[0]['foo'] = old_foo
        assert G._node == H._node

    def test_to_undirected(self):
        G = self.K3
        self.add_attributes(G)
        H = nx.MultiGraph(G)
        self.is_shallow_copy(H, G)
        H = G.to_undirected()
        self.is_deepcopy(H, G)

    def test_to_directed(self):
        G = self.K3
        self.add_attributes(G)
        H = nx.MultiDiGraph(G)
        self.is_shallow_copy(H, G)
        H = G.to_directed()
        self.is_deepcopy(H, G)

    def test_number_of_edges_selfloops(self):
        G = self.K3
        G.add_edge(0, 0)
        G.add_edge(0, 0)
        G.add_edge(0, 0, key='parallel edge')
        G.remove_edge(0, 0, key='parallel edge')
        assert G.number_of_edges(0, 0) == 2
        G.remove_edge(0, 0)
        assert G.number_of_edges(0, 0) == 1

    def test_edge_lookup(self):
        G = self.Graph()
        G.add_edge(1, 2, foo='bar')
        G.add_edge(1, 2, 'key', foo='biz')
        assert edges_equal(G.edges[1, 2, 0], {'foo': 'bar'})
        assert edges_equal(G.edges[1, 2, 'key'], {'foo': 'biz'})

    def test_edge_attr(self):
        G = self.Graph()
        G.add_edge(1, 2, key='k1', foo='bar')
        G.add_edge(1, 2, key='k2', foo='baz')
        assert isinstance(G.get_edge_data(1, 2), G.edge_key_dict_factory)
        assert all((isinstance(d, G.edge_attr_dict_factory) for u, v, d in G.edges(data=True)))
        assert edges_equal(G.edges(keys=True, data=True), [(1, 2, 'k1', {'foo': 'bar'}), (1, 2, 'k2', {'foo': 'baz'})])
        assert edges_equal(G.edges(keys=True, data='foo'), [(1, 2, 'k1', 'bar'), (1, 2, 'k2', 'baz')])

    def test_edge_attr4(self):
        G = self.Graph()
        G.add_edge(1, 2, key=0, data=7, spam='bar', bar='foo')
        assert edges_equal(G.edges(data=True), [(1, 2, {'data': 7, 'spam': 'bar', 'bar': 'foo'})])
        G[1][2][0]['data'] = 10
        assert edges_equal(G.edges(data=True), [(1, 2, {'data': 10, 'spam': 'bar', 'bar': 'foo'})])
        G.adj[1][2][0]['data'] = 20
        assert edges_equal(G.edges(data=True), [(1, 2, {'data': 20, 'spam': 'bar', 'bar': 'foo'})])
        G.edges[1, 2, 0]['data'] = 21
        assert edges_equal(G.edges(data=True), [(1, 2, {'data': 21, 'spam': 'bar', 'bar': 'foo'})])
        G.adj[1][2][0]['listdata'] = [20, 200]
        G.adj[1][2][0]['weight'] = 20
        assert edges_equal(G.edges(data=True), [(1, 2, {'data': 21, 'spam': 'bar', 'bar': 'foo', 'listdata': [20, 200], 'weight': 20})])