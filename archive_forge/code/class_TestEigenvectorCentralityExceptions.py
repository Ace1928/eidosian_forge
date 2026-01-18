import math
import pytest
import networkx as nx
class TestEigenvectorCentralityExceptions:

    def test_multigraph(self):
        with pytest.raises(nx.NetworkXException):
            nx.eigenvector_centrality(nx.MultiGraph())

    def test_multigraph_numpy(self):
        with pytest.raises(nx.NetworkXException):
            nx.eigenvector_centrality_numpy(nx.MultiGraph())

    def test_empty(self):
        with pytest.raises(nx.NetworkXException):
            nx.eigenvector_centrality(nx.Graph())

    def test_empty_numpy(self):
        with pytest.raises(nx.NetworkXException):
            nx.eigenvector_centrality_numpy(nx.Graph())

    def test_zero_nstart(self):
        G = nx.Graph([(1, 2), (1, 3), (2, 3)])
        with pytest.raises(nx.NetworkXException, match='initial vector cannot have all zero values'):
            nx.eigenvector_centrality(G, nstart={v: 0 for v in G})