import collections
import pytest
import networkx as nx
class TestIsEulerian:

    def test_is_eulerian(self):
        assert nx.is_eulerian(nx.complete_graph(5))
        assert nx.is_eulerian(nx.complete_graph(7))
        assert nx.is_eulerian(nx.hypercube_graph(4))
        assert nx.is_eulerian(nx.hypercube_graph(6))
        assert not nx.is_eulerian(nx.complete_graph(4))
        assert not nx.is_eulerian(nx.complete_graph(6))
        assert not nx.is_eulerian(nx.hypercube_graph(3))
        assert not nx.is_eulerian(nx.hypercube_graph(5))
        assert not nx.is_eulerian(nx.petersen_graph())
        assert not nx.is_eulerian(nx.path_graph(4))

    def test_is_eulerian2(self):
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3])
        assert not nx.is_eulerian(G)
        G = nx.DiGraph()
        G.add_nodes_from([1, 2, 3])
        assert not nx.is_eulerian(G)
        G = nx.MultiDiGraph()
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(2, 3)
        G.add_edge(3, 1)
        assert not nx.is_eulerian(G)