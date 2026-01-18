import pytest
import networkx as nx
class TestGroupDegreeCentrality:

    def test_group_degree_centrality_single_node(self):
        """
        Group degree centrality for a single node group
        """
        G = nx.path_graph(4)
        d = nx.group_degree_centrality(G, [1])
        d_answer = nx.degree_centrality(G)[1]
        assert d == d_answer

    def test_group_degree_centrality_multiple_node(self):
        """
        Group degree centrality for group with more than
        1 node
        """
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])
        G.add_edges_from([(1, 2), (1, 3), (1, 6), (1, 7), (1, 8), (2, 3), (2, 4), (2, 5)])
        d = nx.group_degree_centrality(G, [1, 2])
        d_answer = 1
        assert d == d_answer

    def test_group_in_degree_centrality(self):
        """
        Group in-degree centrality in a DiGraph
        """
        G = nx.DiGraph()
        G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])
        G.add_edges_from([(1, 2), (1, 3), (1, 6), (1, 7), (1, 8), (2, 3), (2, 4), (2, 5)])
        d = nx.group_in_degree_centrality(G, [1, 2])
        d_answer = 0
        assert d == d_answer

    def test_group_out_degree_centrality(self):
        """
        Group out-degree centrality in a DiGraph
        """
        G = nx.DiGraph()
        G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])
        G.add_edges_from([(1, 2), (1, 3), (1, 6), (1, 7), (1, 8), (2, 3), (2, 4), (2, 5)])
        d = nx.group_out_degree_centrality(G, [1, 2])
        d_answer = 1
        assert d == d_answer

    def test_group_degree_centrality_node_not_in_graph(self):
        """
        Node(s) in S not in graph, raises NetworkXError
        """
        with pytest.raises(nx.NetworkXError):
            nx.group_degree_centrality(nx.path_graph(5), [6, 7, 8])