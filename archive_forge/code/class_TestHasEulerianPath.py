import collections
import pytest
import networkx as nx
class TestHasEulerianPath:

    def test_has_eulerian_path_cyclic(self):
        assert nx.has_eulerian_path(nx.complete_graph(5))
        assert nx.has_eulerian_path(nx.complete_graph(7))
        assert nx.has_eulerian_path(nx.hypercube_graph(4))
        assert nx.has_eulerian_path(nx.hypercube_graph(6))

    def test_has_eulerian_path_non_cyclic(self):
        assert nx.has_eulerian_path(nx.path_graph(4))
        G = nx.path_graph(6, create_using=nx.DiGraph)
        assert nx.has_eulerian_path(G)

    def test_has_eulerian_path_directed_graph(self):
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (0, 2)])
        assert not nx.has_eulerian_path(G)
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        assert nx.has_eulerian_path(G)
        G.add_node(3)
        assert not nx.has_eulerian_path(G)

    @pytest.mark.parametrize('G', (nx.Graph(), nx.DiGraph()))
    def test_has_eulerian_path_not_weakly_connected(self, G):
        G.add_edges_from([(0, 1), (2, 3), (3, 2)])
        assert not nx.has_eulerian_path(G)

    @pytest.mark.parametrize('G', (nx.Graph(), nx.DiGraph()))
    def test_has_eulerian_path_unbalancedins_more_than_one(self, G):
        G.add_edges_from([(0, 1), (2, 3)])
        assert not nx.has_eulerian_path(G)