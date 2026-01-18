import pytest
import networkx as nx
from networkx.utils import edges_equal
class TestEdgeSubGraph:

    @classmethod
    def setup_class(cls):
        cls.G = G = nx.path_graph(5)
        for i in range(5):
            G.nodes[i]['name'] = f'node{i}'
        G.edges[0, 1]['name'] = 'edge01'
        G.edges[3, 4]['name'] = 'edge34'
        G.graph['name'] = 'graph'
        cls.H = nx.edge_subgraph(G, [(0, 1), (3, 4)])

    def test_correct_nodes(self):
        """Tests that the subgraph has the correct nodes."""
        assert [(0, 'node0'), (1, 'node1'), (3, 'node3'), (4, 'node4')] == sorted(self.H.nodes.data('name'))

    def test_correct_edges(self):
        """Tests that the subgraph has the correct edges."""
        assert edges_equal([(0, 1, 'edge01'), (3, 4, 'edge34')], self.H.edges.data('name'))

    def test_add_node(self):
        """Tests that adding a node to the original graph does not
        affect the nodes of the subgraph.

        """
        self.G.add_node(5)
        assert [0, 1, 3, 4] == sorted(self.H.nodes)
        self.G.remove_node(5)

    def test_remove_node(self):
        """Tests that removing a node in the original graph
        removes the nodes of the subgraph.

        """
        self.G.remove_node(0)
        assert [1, 3, 4] == sorted(self.H.nodes)
        self.G.add_node(0, name='node0')
        self.G.add_edge(0, 1, name='edge01')

    def test_node_attr_dict(self):
        """Tests that the node attribute dictionary of the two graphs is
        the same object.

        """
        for v in self.H:
            assert self.G.nodes[v] == self.H.nodes[v]
        self.G.nodes[0]['name'] = 'foo'
        assert self.G.nodes[0] == self.H.nodes[0]
        self.H.nodes[1]['name'] = 'bar'
        assert self.G.nodes[1] == self.H.nodes[1]
        self.G.nodes[0]['name'] = 'node0'
        self.H.nodes[1]['name'] = 'node1'

    def test_edge_attr_dict(self):
        """Tests that the edge attribute dictionary of the two graphs is
        the same object.

        """
        for u, v in self.H.edges():
            assert self.G.edges[u, v] == self.H.edges[u, v]
        self.G.edges[0, 1]['name'] = 'foo'
        assert self.G.edges[0, 1]['name'] == self.H.edges[0, 1]['name']
        self.H.edges[3, 4]['name'] = 'bar'
        assert self.G.edges[3, 4]['name'] == self.H.edges[3, 4]['name']
        self.G.edges[0, 1]['name'] = 'edge01'
        self.H.edges[3, 4]['name'] = 'edge34'

    def test_graph_attr_dict(self):
        """Tests that the graph attribute dictionary of the two graphs
        is the same object.

        """
        assert self.G.graph is self.H.graph

    def test_readonly(self):
        """Tests that the subgraph cannot change the graph structure"""
        pytest.raises(nx.NetworkXError, self.H.add_node, 5)
        pytest.raises(nx.NetworkXError, self.H.remove_node, 0)
        pytest.raises(nx.NetworkXError, self.H.add_edge, 5, 6)
        pytest.raises(nx.NetworkXError, self.H.remove_edge, 0, 1)