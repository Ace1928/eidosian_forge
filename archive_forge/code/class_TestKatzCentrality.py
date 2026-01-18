import math
import pytest
import networkx as nx
class TestKatzCentrality:

    def test_K5(self):
        """Katz centrality: K5"""
        G = nx.complete_graph(5)
        alpha = 0.1
        b = nx.katz_centrality(G, alpha)
        v = math.sqrt(1 / 5.0)
        b_answer = dict.fromkeys(G, v)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)
        nstart = {n: 1 for n in G}
        b = nx.katz_centrality(G, alpha, nstart=nstart)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_P3(self):
        """Katz centrality: P3"""
        alpha = 0.1
        G = nx.path_graph(3)
        b_answer = {0: 0.5598852584152165, 1: 0.6107839182711449, 2: 0.5598852584152162}
        b = nx.katz_centrality(G, alpha)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=0.0001)

    def test_maxiter(self):
        with pytest.raises(nx.PowerIterationFailedConvergence):
            nx.katz_centrality(nx.path_graph(3), 0.1, max_iter=0)

    def test_beta_as_scalar(self):
        alpha = 0.1
        beta = 0.1
        b_answer = {0: 0.5598852584152165, 1: 0.6107839182711449, 2: 0.5598852584152162}
        G = nx.path_graph(3)
        b = nx.katz_centrality(G, alpha, beta)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=0.0001)

    def test_beta_as_dict(self):
        alpha = 0.1
        beta = {0: 1.0, 1: 1.0, 2: 1.0}
        b_answer = {0: 0.5598852584152165, 1: 0.6107839182711449, 2: 0.5598852584152162}
        G = nx.path_graph(3)
        b = nx.katz_centrality(G, alpha, beta)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=0.0001)

    def test_multiple_alpha(self):
        alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        for alpha in alpha_list:
            b_answer = {0.1: {0: 0.5598852584152165, 1: 0.6107839182711449, 2: 0.5598852584152162}, 0.2: {0: 0.5454545454545454, 1: 0.6363636363636365, 2: 0.5454545454545454}, 0.3: {0: 0.5333964609104419, 1: 0.6564879518897746, 2: 0.5333964609104419}, 0.4: {0: 0.5232045649263551, 1: 0.6726915834767423, 2: 0.5232045649263551}, 0.5: {0: 0.5144957746691622, 1: 0.6859943117075809, 2: 0.5144957746691622}, 0.6: {0: 0.5069794004195823, 1: 0.6970966755769258, 2: 0.5069794004195823}}
            G = nx.path_graph(3)
            b = nx.katz_centrality(G, alpha)
            for n in sorted(G):
                assert b[n] == pytest.approx(b_answer[alpha][n], abs=0.0001)

    def test_multigraph(self):
        with pytest.raises(nx.NetworkXException):
            nx.katz_centrality(nx.MultiGraph(), 0.1)

    def test_empty(self):
        e = nx.katz_centrality(nx.Graph(), 0.1)
        assert e == {}

    def test_bad_beta(self):
        with pytest.raises(nx.NetworkXException):
            G = nx.Graph([(0, 1)])
            beta = {0: 77}
            nx.katz_centrality(G, 0.1, beta=beta)

    def test_bad_beta_number(self):
        with pytest.raises(nx.NetworkXException):
            G = nx.Graph([(0, 1)])
            nx.katz_centrality(G, 0.1, beta='foo')