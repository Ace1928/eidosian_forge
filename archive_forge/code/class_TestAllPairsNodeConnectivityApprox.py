import pytest
import networkx as nx
from networkx.algorithms import approximation as approx
class TestAllPairsNodeConnectivityApprox:

    @classmethod
    def setup_class(cls):
        cls.path = nx.path_graph(7)
        cls.directed_path = nx.path_graph(7, create_using=nx.DiGraph())
        cls.cycle = nx.cycle_graph(7)
        cls.directed_cycle = nx.cycle_graph(7, create_using=nx.DiGraph())
        cls.gnp = nx.gnp_random_graph(30, 0.1)
        cls.directed_gnp = nx.gnp_random_graph(30, 0.1, directed=True)
        cls.K20 = nx.complete_graph(20)
        cls.K10 = nx.complete_graph(10)
        cls.K5 = nx.complete_graph(5)
        cls.G_list = [cls.path, cls.directed_path, cls.cycle, cls.directed_cycle, cls.gnp, cls.directed_gnp, cls.K10, cls.K5, cls.K20]

    def test_cycles(self):
        K_undir = approx.all_pairs_node_connectivity(self.cycle)
        for source in K_undir:
            for target, k in K_undir[source].items():
                assert k == 2
        K_dir = approx.all_pairs_node_connectivity(self.directed_cycle)
        for source in K_dir:
            for target, k in K_dir[source].items():
                assert k == 1

    def test_complete(self):
        for G in [self.K10, self.K5, self.K20]:
            K = approx.all_pairs_node_connectivity(G)
            for source in K:
                for target, k in K[source].items():
                    assert k == len(G) - 1

    def test_paths(self):
        K_undir = approx.all_pairs_node_connectivity(self.path)
        for source in K_undir:
            for target, k in K_undir[source].items():
                assert k == 1
        K_dir = approx.all_pairs_node_connectivity(self.directed_path)
        for source in K_dir:
            for target, k in K_dir[source].items():
                if source < target:
                    assert k == 1
                else:
                    assert k == 0

    def test_cutoff(self):
        for G in [self.K10, self.K5, self.K20]:
            for mp in [2, 3, 4]:
                paths = approx.all_pairs_node_connectivity(G, cutoff=mp)
                for source in paths:
                    for target, K in paths[source].items():
                        assert K == mp

    def test_all_pairs_connectivity_nbunch(self):
        G = nx.complete_graph(5)
        nbunch = [0, 2, 3]
        C = approx.all_pairs_node_connectivity(G, nbunch=nbunch)
        assert len(C) == len(nbunch)