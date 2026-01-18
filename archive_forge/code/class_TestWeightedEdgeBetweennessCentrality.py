import pytest
import networkx as nx
class TestWeightedEdgeBetweennessCentrality:

    def test_K5(self):
        """Edge betweenness centrality: K5"""
        G = nx.complete_graph(5)
        b = nx.edge_betweenness_centrality(G, weight='weight', normalized=False)
        b_answer = dict.fromkeys(G.edges(), 1)
        for n in sorted(G.edges()):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_C4(self):
        """Edge betweenness centrality: C4"""
        G = nx.cycle_graph(4)
        b = nx.edge_betweenness_centrality(G, weight='weight', normalized=False)
        b_answer = {(0, 1): 2, (0, 3): 2, (1, 2): 2, (2, 3): 2}
        for n in sorted(G.edges()):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_P4(self):
        """Edge betweenness centrality: P4"""
        G = nx.path_graph(4)
        b = nx.edge_betweenness_centrality(G, weight='weight', normalized=False)
        b_answer = {(0, 1): 3, (1, 2): 4, (2, 3): 3}
        for n in sorted(G.edges()):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_balanced_tree(self):
        """Edge betweenness centrality: balanced tree"""
        G = nx.balanced_tree(r=2, h=2)
        b = nx.edge_betweenness_centrality(G, weight='weight', normalized=False)
        b_answer = {(0, 1): 12, (0, 2): 12, (1, 3): 6, (1, 4): 6, (2, 5): 6, (2, 6): 6}
        for n in sorted(G.edges()):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_weighted_graph(self):
        """Edge betweenness centrality: weighted"""
        eList = [(0, 1, 5), (0, 2, 4), (0, 3, 3), (0, 4, 2), (1, 2, 4), (1, 3, 1), (1, 4, 3), (2, 4, 5), (3, 4, 4)]
        G = nx.Graph()
        G.add_weighted_edges_from(eList)
        b = nx.edge_betweenness_centrality(G, weight='weight', normalized=False)
        b_answer = {(0, 1): 0.0, (0, 2): 1.0, (0, 3): 2.0, (0, 4): 1.0, (1, 2): 2.0, (1, 3): 3.5, (1, 4): 1.5, (2, 4): 1.0, (3, 4): 0.5}
        for n in sorted(G.edges()):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_normalized_weighted_graph(self):
        """Edge betweenness centrality: normalized weighted"""
        eList = [(0, 1, 5), (0, 2, 4), (0, 3, 3), (0, 4, 2), (1, 2, 4), (1, 3, 1), (1, 4, 3), (2, 4, 5), (3, 4, 4)]
        G = nx.Graph()
        G.add_weighted_edges_from(eList)
        b = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)
        b_answer = {(0, 1): 0.0, (0, 2): 1.0, (0, 3): 2.0, (0, 4): 1.0, (1, 2): 2.0, (1, 3): 3.5, (1, 4): 1.5, (2, 4): 1.0, (3, 4): 0.5}
        norm = len(G) * (len(G) - 1) / 2
        for n in sorted(G.edges()):
            assert b[n] == pytest.approx(b_answer[n] / norm, abs=1e-07)

    def test_weighted_multigraph(self):
        """Edge betweenness centrality: weighted multigraph"""
        eList = [(0, 1, 5), (0, 1, 4), (0, 2, 4), (0, 3, 3), (0, 3, 3), (0, 4, 2), (1, 2, 4), (1, 3, 1), (1, 3, 2), (1, 4, 3), (1, 4, 4), (2, 4, 5), (3, 4, 4), (3, 4, 4)]
        G = nx.MultiGraph()
        G.add_weighted_edges_from(eList)
        b = nx.edge_betweenness_centrality(G, weight='weight', normalized=False)
        b_answer = {(0, 1, 0): 0.0, (0, 1, 1): 0.5, (0, 2, 0): 1.0, (0, 3, 0): 0.75, (0, 3, 1): 0.75, (0, 4, 0): 1.0, (1, 2, 0): 2.0, (1, 3, 0): 3.0, (1, 3, 1): 0.0, (1, 4, 0): 1.5, (1, 4, 1): 0.0, (2, 4, 0): 1.0, (3, 4, 0): 0.25, (3, 4, 1): 0.25}
        for n in sorted(G.edges(keys=True)):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_normalized_weighted_multigraph(self):
        """Edge betweenness centrality: normalized weighted multigraph"""
        eList = [(0, 1, 5), (0, 1, 4), (0, 2, 4), (0, 3, 3), (0, 3, 3), (0, 4, 2), (1, 2, 4), (1, 3, 1), (1, 3, 2), (1, 4, 3), (1, 4, 4), (2, 4, 5), (3, 4, 4), (3, 4, 4)]
        G = nx.MultiGraph()
        G.add_weighted_edges_from(eList)
        b = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)
        b_answer = {(0, 1, 0): 0.0, (0, 1, 1): 0.5, (0, 2, 0): 1.0, (0, 3, 0): 0.75, (0, 3, 1): 0.75, (0, 4, 0): 1.0, (1, 2, 0): 2.0, (1, 3, 0): 3.0, (1, 3, 1): 0.0, (1, 4, 0): 1.5, (1, 4, 1): 0.0, (2, 4, 0): 1.0, (3, 4, 0): 0.25, (3, 4, 1): 0.25}
        norm = len(G) * (len(G) - 1) / 2
        for n in sorted(G.edges(keys=True)):
            assert b[n] == pytest.approx(b_answer[n] / norm, abs=1e-07)