from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
class TestMinimumCycleBasis:

    @classmethod
    def setup_class(cls):
        T = nx.Graph()
        nx.add_cycle(T, [1, 2, 3, 4], weight=1)
        T.add_edge(2, 4, weight=5)
        cls.diamond_graph = T

    def test_unweighted_diamond(self):
        mcb = nx.minimum_cycle_basis(self.diamond_graph)
        assert_basis_equal(mcb, [[2, 4, 1], [3, 4, 2]])

    def test_weighted_diamond(self):
        mcb = nx.minimum_cycle_basis(self.diamond_graph, weight='weight')
        assert_basis_equal(mcb, [[2, 4, 1], [4, 3, 2, 1]])

    def test_dimensionality(self):
        ntrial = 10
        for seed in range(1234, 1234 + ntrial):
            rg = nx.erdos_renyi_graph(10, 0.3, seed=seed)
            nnodes = rg.number_of_nodes()
            nedges = rg.number_of_edges()
            ncomp = nx.number_connected_components(rg)
            mcb = nx.minimum_cycle_basis(rg)
            assert len(mcb) == nedges - nnodes + ncomp
            check_independent(mcb)

    def test_complete_graph(self):
        cg = nx.complete_graph(5)
        mcb = nx.minimum_cycle_basis(cg)
        assert all((len(cycle) == 3 for cycle in mcb))
        check_independent(mcb)

    def test_tree_graph(self):
        tg = nx.balanced_tree(3, 3)
        assert not nx.minimum_cycle_basis(tg)

    def test_petersen_graph(self):
        G = nx.petersen_graph()
        mcb = list(nx.minimum_cycle_basis(G))
        expected = [[4, 9, 7, 5, 0], [1, 2, 3, 4, 0], [1, 6, 8, 5, 0], [4, 3, 8, 5, 0], [1, 6, 9, 4, 0], [1, 2, 7, 5, 0]]
        assert len(mcb) == len(expected)
        assert all((c in expected for c in mcb))
        for c in mcb:
            assert all((G.has_edge(u, v) for u, v in nx.utils.pairwise(c, cyclic=True)))
        check_independent(mcb)

    def test_gh6787_variable_weighted_complete_graph(self):
        N = 8
        cg = nx.complete_graph(N)
        cg.add_weighted_edges_from([(u, v, 9) for u, v in cg.edges])
        cg.add_weighted_edges_from([(u, v, 1) for u, v in nx.cycle_graph(N).edges])
        mcb = nx.minimum_cycle_basis(cg, weight='weight')
        check_independent(mcb)

    def test_gh6787_and_edge_attribute_names(self):
        G = nx.cycle_graph(4)
        G.add_weighted_edges_from([(0, 2, 10), (1, 3, 10)], weight='dist')
        expected = [[1, 3, 0], [3, 2, 1, 0], [1, 2, 0]]
        mcb = list(nx.minimum_cycle_basis(G, weight='dist'))
        assert len(mcb) == len(expected)
        assert all((c in expected for c in mcb))
        expected = [[1, 3, 0], [1, 2, 0], [3, 2, 0]]
        mcb = list(nx.minimum_cycle_basis(G))
        assert len(mcb) == len(expected)
        assert all((c in expected for c in mcb))