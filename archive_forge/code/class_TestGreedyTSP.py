import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
class TestGreedyTSP(TestBase):

    def test_greedy(self):
        cycle = nx_app.greedy_tsp(self.DG, source='D')
        cost = sum((self.DG[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
        validate_solution(cycle, cost, ['D', 'C', 'B', 'A', 'D'], 31.0)
        cycle = nx_app.greedy_tsp(self.DG2, source='D')
        cost = sum((self.DG2[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
        validate_solution(cycle, cost, ['D', 'C', 'B', 'A', 'D'], 78.0)
        cycle = nx_app.greedy_tsp(self.UG, source='D')
        cost = sum((self.UG[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
        validate_solution(cycle, cost, ['D', 'C', 'B', 'A', 'D'], 33.0)
        cycle = nx_app.greedy_tsp(self.UG2, source='D')
        cost = sum((self.UG2[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
        validate_solution(cycle, cost, ['D', 'C', 'A', 'B', 'D'], 27.0)

    def test_not_complete_graph(self):
        pytest.raises(nx.NetworkXError, nx_app.greedy_tsp, self.incompleteUG)
        pytest.raises(nx.NetworkXError, nx_app.greedy_tsp, self.incompleteDG)

    def test_not_weighted_graph(self):
        nx_app.greedy_tsp(self.unweightedUG)
        nx_app.greedy_tsp(self.unweightedDG)

    def test_two_nodes(self):
        G = nx.Graph()
        G.add_weighted_edges_from({(1, 2, 1)})
        cycle = nx_app.greedy_tsp(G)
        cost = sum((G[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
        validate_solution(cycle, cost, [1, 2, 1], 2)

    def test_ignore_selfloops(self):
        G = nx.complete_graph(5)
        G.add_edge(3, 3)
        cycle = nx_app.greedy_tsp(G)
        assert len(cycle) - 1 == len(G) == len(set(cycle))