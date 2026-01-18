import pytest
import networkx as nx
from networkx import approximate_current_flow_betweenness_centrality as approximate_cfbc
from networkx import edge_current_flow_betweenness_centrality as edge_current_flow
class TestFlowBetweennessCentrality:

    def test_K4_normalized(self):
        """Betweenness centrality: K4"""
        G = nx.complete_graph(4)
        b = nx.current_flow_betweenness_centrality(G, normalized=True)
        b_answer = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)
        G.add_edge(0, 1, weight=0.5, other=0.3)
        b = nx.current_flow_betweenness_centrality(G, normalized=True, weight=None)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)
        wb_answer = {0: 0.2222222, 1: 0.2222222, 2: 0.30555555, 3: 0.30555555}
        b = nx.current_flow_betweenness_centrality(G, normalized=True, weight='weight')
        for n in sorted(G):
            assert b[n] == pytest.approx(wb_answer[n], abs=1e-07)
        wb_answer = {0: 0.2051282, 1: 0.2051282, 2: 0.33974358, 3: 0.33974358}
        b = nx.current_flow_betweenness_centrality(G, normalized=True, weight='other')
        for n in sorted(G):
            assert b[n] == pytest.approx(wb_answer[n], abs=1e-07)

    def test_K4(self):
        """Betweenness centrality: K4"""
        G = nx.complete_graph(4)
        for solver in ['full', 'lu', 'cg']:
            b = nx.current_flow_betweenness_centrality(G, normalized=False, solver=solver)
            b_answer = {0: 0.75, 1: 0.75, 2: 0.75, 3: 0.75}
            for n in sorted(G):
                assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_P4_normalized(self):
        """Betweenness centrality: P4 normalized"""
        G = nx.path_graph(4)
        b = nx.current_flow_betweenness_centrality(G, normalized=True)
        b_answer = {0: 0, 1: 2.0 / 3, 2: 2.0 / 3, 3: 0}
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_P4(self):
        """Betweenness centrality: P4"""
        G = nx.path_graph(4)
        b = nx.current_flow_betweenness_centrality(G, normalized=False)
        b_answer = {0: 0, 1: 2, 2: 2, 3: 0}
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_star(self):
        """Betweenness centrality: star"""
        G = nx.Graph()
        nx.add_star(G, ['a', 'b', 'c', 'd'])
        b = nx.current_flow_betweenness_centrality(G, normalized=True)
        b_answer = {'a': 1.0, 'b': 0.0, 'c': 0.0, 'd': 0.0}
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_solvers2(self):
        """Betweenness centrality: alternate solvers"""
        G = nx.complete_graph(4)
        for solver in ['full', 'lu', 'cg']:
            b = nx.current_flow_betweenness_centrality(G, normalized=False, solver=solver)
            b_answer = {0: 0.75, 1: 0.75, 2: 0.75, 3: 0.75}
            for n in sorted(G):
                assert b[n] == pytest.approx(b_answer[n], abs=1e-07)