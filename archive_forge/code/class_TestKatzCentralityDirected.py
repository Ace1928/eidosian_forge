import math
import pytest
import networkx as nx
class TestKatzCentralityDirected:

    @classmethod
    def setup_class(cls):
        G = nx.DiGraph()
        edges = [(1, 2), (1, 3), (2, 4), (3, 2), (3, 5), (4, 2), (4, 5), (4, 6), (5, 6), (5, 7), (5, 8), (6, 8), (7, 1), (7, 5), (7, 8), (8, 6), (8, 7)]
        G.add_edges_from(edges, weight=2.0)
        cls.G = G.reverse()
        cls.G.alpha = 0.1
        cls.G.evc = [0.3289589783189635, 0.2832077296243516, 0.3425906003685471, 0.3970420865198392, 0.41074871061646284, 0.272257430756461, 0.4201989685435462, 0.34229059218038554]
        H = nx.DiGraph(edges)
        cls.H = G.reverse()
        cls.H.alpha = 0.1
        cls.H.evc = [0.3289589783189635, 0.2832077296243516, 0.3425906003685471, 0.3970420865198392, 0.41074871061646284, 0.272257430756461, 0.4201989685435462, 0.34229059218038554]

    def test_katz_centrality_weighted(self):
        G = self.G
        alpha = self.G.alpha
        p = nx.katz_centrality(G, alpha, weight='weight')
        for a, b in zip(list(p.values()), self.G.evc):
            assert a == pytest.approx(b, abs=1e-07)

    def test_katz_centrality_unweighted(self):
        H = self.H
        alpha = self.H.alpha
        p = nx.katz_centrality(H, alpha, weight='weight')
        for a, b in zip(list(p.values()), self.H.evc):
            assert a == pytest.approx(b, abs=1e-07)