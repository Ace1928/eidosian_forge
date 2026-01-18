import pytest
import networkx as nx
class TestLocalReachingCentrality:
    """Unit tests for the local reaching centrality function."""

    def test_non_positive_weights(self):
        with pytest.raises(nx.NetworkXError):
            G = nx.DiGraph()
            G.add_weighted_edges_from([(0, 1, 0)])
            nx.local_reaching_centrality(G, 0, weight='weight')

    def test_negatively_weighted(self):
        with pytest.raises(nx.NetworkXError):
            G = nx.Graph()
            G.add_weighted_edges_from([(0, 1, -2), (1, 2, +1)])
            nx.local_reaching_centrality(G, 0, weight='weight')

    def test_undirected_unweighted_star(self):
        G = nx.star_graph(2)
        grc = nx.local_reaching_centrality
        assert grc(G, 1, weight=None, normalized=False) == 0.75

    def test_undirected_weighted_star(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(1, 2, 1), (1, 3, 2)])
        centrality = nx.local_reaching_centrality(G, 1, normalized=False, weight='weight')
        assert centrality == 1.5

    def test_undirected_weighted_normalized(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(1, 2, 1), (1, 3, 2)])
        centrality = nx.local_reaching_centrality(G, 1, normalized=True, weight='weight')
        assert centrality == 1.0