import collections
import pytest
import networkx as nx
class TestEulerianCircuit:

    def test_eulerian_circuit_cycle(self):
        G = nx.cycle_graph(4)
        edges = list(nx.eulerian_circuit(G, source=0))
        nodes = [u for u, v in edges]
        assert nodes == [0, 3, 2, 1]
        assert edges == [(0, 3), (3, 2), (2, 1), (1, 0)]
        edges = list(nx.eulerian_circuit(G, source=1))
        nodes = [u for u, v in edges]
        assert nodes == [1, 2, 3, 0]
        assert edges == [(1, 2), (2, 3), (3, 0), (0, 1)]
        G = nx.complete_graph(3)
        edges = list(nx.eulerian_circuit(G, source=0))
        nodes = [u for u, v in edges]
        assert nodes == [0, 2, 1]
        assert edges == [(0, 2), (2, 1), (1, 0)]
        edges = list(nx.eulerian_circuit(G, source=1))
        nodes = [u for u, v in edges]
        assert nodes == [1, 2, 0]
        assert edges == [(1, 2), (2, 0), (0, 1)]

    def test_eulerian_circuit_digraph(self):
        G = nx.DiGraph()
        nx.add_cycle(G, [0, 1, 2, 3])
        edges = list(nx.eulerian_circuit(G, source=0))
        nodes = [u for u, v in edges]
        assert nodes == [0, 1, 2, 3]
        assert edges == [(0, 1), (1, 2), (2, 3), (3, 0)]
        edges = list(nx.eulerian_circuit(G, source=1))
        nodes = [u for u, v in edges]
        assert nodes == [1, 2, 3, 0]
        assert edges == [(1, 2), (2, 3), (3, 0), (0, 1)]

    def test_multigraph(self):
        G = nx.MultiGraph()
        nx.add_cycle(G, [0, 1, 2, 3])
        G.add_edge(1, 2)
        G.add_edge(1, 2)
        edges = list(nx.eulerian_circuit(G, source=0))
        nodes = [u for u, v in edges]
        assert nodes == [0, 3, 2, 1, 2, 1]
        assert edges == [(0, 3), (3, 2), (2, 1), (1, 2), (2, 1), (1, 0)]

    def test_multigraph_with_keys(self):
        G = nx.MultiGraph()
        nx.add_cycle(G, [0, 1, 2, 3])
        G.add_edge(1, 2)
        G.add_edge(1, 2)
        edges = list(nx.eulerian_circuit(G, source=0, keys=True))
        nodes = [u for u, v, k in edges]
        assert nodes == [0, 3, 2, 1, 2, 1]
        assert edges[:2] == [(0, 3, 0), (3, 2, 0)]
        assert collections.Counter(edges[2:5]) == collections.Counter([(2, 1, 0), (1, 2, 1), (2, 1, 2)])
        assert edges[5:] == [(1, 0, 0)]

    def test_not_eulerian(self):
        with pytest.raises(nx.NetworkXError):
            f = list(nx.eulerian_circuit(nx.complete_graph(4)))