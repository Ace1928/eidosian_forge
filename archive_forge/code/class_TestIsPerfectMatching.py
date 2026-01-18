import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
class TestIsPerfectMatching:
    """Unit tests for the
    :func:`~networkx.algorithms.matching.is_perfect_matching` function.

    """

    def test_dict(self):
        G = nx.path_graph(4)
        assert nx.is_perfect_matching(G, {0: 1, 1: 0, 2: 3, 3: 2})

    def test_valid(self):
        G = nx.path_graph(4)
        assert nx.is_perfect_matching(G, {(0, 1), (2, 3)})

    def test_valid_not_path(self):
        G = nx.cycle_graph(4)
        G.add_edge(0, 4)
        G.add_edge(1, 4)
        G.add_edge(5, 2)
        assert nx.is_perfect_matching(G, {(1, 4), (0, 3), (5, 2)})

    def test_invalid_input(self):
        error = nx.NetworkXError
        G = nx.path_graph(4)
        raises(error, nx.is_perfect_matching, G, {(0, 5)})
        raises(error, nx.is_perfect_matching, G, {(5, 0)})
        raises(error, nx.is_perfect_matching, G, {(0, 1, 2), (2, 3)})
        raises(error, nx.is_perfect_matching, G, {(0,), (2, 3)})

    def test_selfloops(self):
        error = nx.NetworkXError
        G = nx.path_graph(4)
        raises(error, nx.is_perfect_matching, G, {(5, 5), (2, 3)})
        assert not nx.is_perfect_matching(G, {(0, 0), (1, 2), (2, 3)})
        G.add_edge(0, 0)
        assert not nx.is_perfect_matching(G, {(0, 0), (1, 2)})

    def test_not_matching(self):
        G = nx.path_graph(4)
        assert not nx.is_perfect_matching(G, {(0, 3)})
        assert not nx.is_perfect_matching(G, {(0, 1), (1, 2), (2, 3)})

    def test_maximal_but_not_perfect(self):
        G = nx.cycle_graph(4)
        G.add_edge(0, 4)
        G.add_edge(1, 4)
        assert not nx.is_perfect_matching(G, {(1, 4), (0, 3)})