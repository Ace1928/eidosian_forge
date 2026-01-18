import pytest
import networkx as nx
class TestGroupBetweennessCentrality:

    def test_group_betweenness_single_node(self):
        """
        Group betweenness centrality for single node group
        """
        G = nx.path_graph(5)
        C = [1]
        b = nx.group_betweenness_centrality(G, C, weight=None, normalized=False, endpoints=False)
        b_answer = 3.0
        assert b == b_answer

    def test_group_betweenness_with_endpoints(self):
        """
        Group betweenness centrality for single node group
        """
        G = nx.path_graph(5)
        C = [1]
        b = nx.group_betweenness_centrality(G, C, weight=None, normalized=False, endpoints=True)
        b_answer = 7.0
        assert b == b_answer

    def test_group_betweenness_normalized(self):
        """
        Group betweenness centrality for group with more than
        1 node and normalized
        """
        G = nx.path_graph(5)
        C = [1, 3]
        b = nx.group_betweenness_centrality(G, C, weight=None, normalized=True, endpoints=False)
        b_answer = 1.0
        assert b == b_answer

    def test_two_group_betweenness_value_zero(self):
        """
        Group betweenness centrality value of 0
        """
        G = nx.cycle_graph(7)
        C = [[0, 1, 6], [0, 1, 5]]
        b = nx.group_betweenness_centrality(G, C, weight=None, normalized=False)
        b_answer = [0.0, 3.0]
        assert b == b_answer

    def test_group_betweenness_value_zero(self):
        """
        Group betweenness centrality value of 0
        """
        G = nx.cycle_graph(6)
        C = [0, 1, 5]
        b = nx.group_betweenness_centrality(G, C, weight=None, normalized=False)
        b_answer = 0.0
        assert b == b_answer

    def test_group_betweenness_disconnected_graph(self):
        """
        Group betweenness centrality in a disconnected graph
        """
        G = nx.path_graph(5)
        G.remove_edge(0, 1)
        C = [1]
        b = nx.group_betweenness_centrality(G, C, weight=None, normalized=False)
        b_answer = 0.0
        assert b == b_answer

    def test_group_betweenness_node_not_in_graph(self):
        """
        Node(s) in C not in graph, raises NodeNotFound exception
        """
        with pytest.raises(nx.NodeNotFound):
            nx.group_betweenness_centrality(nx.path_graph(5), [4, 7, 8])

    def test_group_betweenness_directed_weighted(self):
        """
        Group betweenness centrality in a directed and weighted graph
        """
        G = nx.DiGraph()
        G.add_edge(1, 0, weight=1)
        G.add_edge(0, 2, weight=2)
        G.add_edge(1, 2, weight=3)
        G.add_edge(3, 1, weight=4)
        G.add_edge(2, 3, weight=1)
        G.add_edge(4, 3, weight=6)
        G.add_edge(2, 4, weight=7)
        C = [1, 2]
        b = nx.group_betweenness_centrality(G, C, weight='weight', normalized=False)
        b_answer = 5.0
        assert b == b_answer