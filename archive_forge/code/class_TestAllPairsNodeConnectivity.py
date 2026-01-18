import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity import (
class TestAllPairsNodeConnectivity:

    @classmethod
    def setup_class(cls):
        cls.path = nx.path_graph(7)
        cls.directed_path = nx.path_graph(7, create_using=nx.DiGraph())
        cls.cycle = nx.cycle_graph(7)
        cls.directed_cycle = nx.cycle_graph(7, create_using=nx.DiGraph())
        cls.gnp = nx.gnp_random_graph(30, 0.1, seed=42)
        cls.directed_gnp = nx.gnp_random_graph(30, 0.1, directed=True, seed=42)
        cls.K20 = nx.complete_graph(20)
        cls.K10 = nx.complete_graph(10)
        cls.K5 = nx.complete_graph(5)
        cls.G_list = [cls.path, cls.directed_path, cls.cycle, cls.directed_cycle, cls.gnp, cls.directed_gnp, cls.K10, cls.K5, cls.K20]

    def test_cycles(self):
        K_undir = nx.all_pairs_node_connectivity(self.cycle)
        for source in K_undir:
            for target, k in K_undir[source].items():
                assert k == 2
        K_dir = nx.all_pairs_node_connectivity(self.directed_cycle)
        for source in K_dir:
            for target, k in K_dir[source].items():
                assert k == 1

    def test_complete(self):
        for G in [self.K10, self.K5, self.K20]:
            K = nx.all_pairs_node_connectivity(G)
            for source in K:
                for target, k in K[source].items():
                    assert k == len(G) - 1

    def test_paths(self):
        K_undir = nx.all_pairs_node_connectivity(self.path)
        for source in K_undir:
            for target, k in K_undir[source].items():
                assert k == 1
        K_dir = nx.all_pairs_node_connectivity(self.directed_path)
        for source in K_dir:
            for target, k in K_dir[source].items():
                if source < target:
                    assert k == 1
                else:
                    assert k == 0

    def test_all_pairs_connectivity_nbunch(self):
        G = nx.complete_graph(5)
        nbunch = [0, 2, 3]
        C = nx.all_pairs_node_connectivity(G, nbunch=nbunch)
        assert len(C) == len(nbunch)

    def test_all_pairs_connectivity_icosahedral(self):
        G = nx.icosahedral_graph()
        C = nx.all_pairs_node_connectivity(G)
        assert all((5 == C[u][v] for u, v in itertools.combinations(G, 2)))

    def test_all_pairs_connectivity(self):
        G = nx.Graph()
        nodes = [0, 1, 2, 3]
        nx.add_path(G, nodes)
        A = {n: {} for n in G}
        for u, v in itertools.combinations(nodes, 2):
            A[u][v] = A[v][u] = nx.node_connectivity(G, u, v)
        C = nx.all_pairs_node_connectivity(G)
        assert sorted(((k, sorted(v)) for k, v in A.items())) == sorted(((k, sorted(v)) for k, v in C.items()))

    def test_all_pairs_connectivity_directed(self):
        G = nx.DiGraph()
        nodes = [0, 1, 2, 3]
        nx.add_path(G, nodes)
        A = {n: {} for n in G}
        for u, v in itertools.permutations(nodes, 2):
            A[u][v] = nx.node_connectivity(G, u, v)
        C = nx.all_pairs_node_connectivity(G)
        assert sorted(((k, sorted(v)) for k, v in A.items())) == sorted(((k, sorted(v)) for k, v in C.items()))

    def test_all_pairs_connectivity_nbunch_combinations(self):
        G = nx.complete_graph(5)
        nbunch = [0, 2, 3]
        A = {n: {} for n in nbunch}
        for u, v in itertools.combinations(nbunch, 2):
            A[u][v] = A[v][u] = nx.node_connectivity(G, u, v)
        C = nx.all_pairs_node_connectivity(G, nbunch=nbunch)
        assert sorted(((k, sorted(v)) for k, v in A.items())) == sorted(((k, sorted(v)) for k, v in C.items()))

    def test_all_pairs_connectivity_nbunch_iter(self):
        G = nx.complete_graph(5)
        nbunch = [0, 2, 3]
        A = {n: {} for n in nbunch}
        for u, v in itertools.combinations(nbunch, 2):
            A[u][v] = A[v][u] = nx.node_connectivity(G, u, v)
        C = nx.all_pairs_node_connectivity(G, nbunch=iter(nbunch))
        assert sorted(((k, sorted(v)) for k, v in A.items())) == sorted(((k, sorted(v)) for k, v in C.items()))