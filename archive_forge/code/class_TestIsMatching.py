import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
class TestIsMatching:
    """Unit tests for the
    :func:`~networkx.algorithms.matching.is_matching` function.

    """

    def test_dict(self):
        G = nx.path_graph(4)
        assert nx.is_matching(G, {0: 1, 1: 0, 2: 3, 3: 2})

    def test_empty_matching(self):
        G = nx.path_graph(4)
        assert nx.is_matching(G, set())

    def test_single_edge(self):
        G = nx.path_graph(4)
        assert nx.is_matching(G, {(1, 2)})

    def test_edge_order(self):
        G = nx.path_graph(4)
        assert nx.is_matching(G, {(0, 1), (2, 3)})
        assert nx.is_matching(G, {(1, 0), (2, 3)})
        assert nx.is_matching(G, {(0, 1), (3, 2)})
        assert nx.is_matching(G, {(1, 0), (3, 2)})

    def test_valid_matching(self):
        G = nx.path_graph(4)
        assert nx.is_matching(G, {(0, 1), (2, 3)})

    def test_invalid_input(self):
        error = nx.NetworkXError
        G = nx.path_graph(4)
        raises(error, nx.is_matching, G, {(0, 5), (2, 3)})
        raises(error, nx.is_matching, G, {(0, 1, 2), (2, 3)})
        raises(error, nx.is_matching, G, {(0,), (2, 3)})

    def test_selfloops(self):
        error = nx.NetworkXError
        G = nx.path_graph(4)
        raises(error, nx.is_matching, G, {(5, 5), (2, 3)})
        assert not nx.is_matching(G, {(0, 0), (1, 2), (2, 3)})
        G.add_edge(0, 0)
        assert not nx.is_matching(G, {(0, 0), (1, 2)})

    def test_invalid_matching(self):
        G = nx.path_graph(4)
        assert not nx.is_matching(G, {(0, 1), (1, 2), (2, 3)})

    def test_invalid_edge(self):
        G = nx.path_graph(4)
        assert not nx.is_matching(G, {(0, 3), (1, 2)})
        raises(nx.NetworkXError, nx.is_matching, G, {(0, 55)})
        G = nx.DiGraph(G.edges)
        assert nx.is_matching(G, {(0, 1)})
        assert not nx.is_matching(G, {(1, 0)})