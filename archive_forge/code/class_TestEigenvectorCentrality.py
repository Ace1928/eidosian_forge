import math
import pytest
import networkx as nx
class TestEigenvectorCentrality:

    def test_K5(self):
        """Eigenvector centrality: K5"""
        G = nx.complete_graph(5)
        b = nx.eigenvector_centrality(G)
        v = math.sqrt(1 / 5.0)
        b_answer = dict.fromkeys(G, v)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)
        nstart = {n: 1 for n in G}
        b = nx.eigenvector_centrality(G, nstart=nstart)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)
        b = nx.eigenvector_centrality_numpy(G)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=0.001)

    def test_P3(self):
        """Eigenvector centrality: P3"""
        G = nx.path_graph(3)
        b_answer = {0: 0.5, 1: 0.7071, 2: 0.5}
        b = nx.eigenvector_centrality_numpy(G)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=0.0001)
        b = nx.eigenvector_centrality(G)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=0.0001)

    def test_P3_unweighted(self):
        """Eigenvector centrality: P3"""
        G = nx.path_graph(3)
        b_answer = {0: 0.5, 1: 0.7071, 2: 0.5}
        b = nx.eigenvector_centrality_numpy(G, weight=None)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=0.0001)

    def test_maxiter(self):
        with pytest.raises(nx.PowerIterationFailedConvergence):
            G = nx.path_graph(3)
            nx.eigenvector_centrality(G, max_iter=0)