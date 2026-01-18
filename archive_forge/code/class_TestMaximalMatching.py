import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
class TestMaximalMatching:
    """Unit tests for the
    :func:`~networkx.algorithms.matching.maximal_matching`.

    """

    def test_valid_matching(self):
        edges = [(1, 2), (1, 5), (2, 3), (2, 5), (3, 4), (3, 6), (5, 6)]
        G = nx.Graph(edges)
        matching = nx.maximal_matching(G)
        assert nx.is_maximal_matching(G, matching)

    def test_single_edge_matching(self):
        G = nx.star_graph(5)
        matching = nx.maximal_matching(G)
        assert 1 == len(matching)
        assert nx.is_maximal_matching(G, matching)

    def test_self_loops(self):
        G = nx.path_graph(3)
        G.add_edges_from([(0, 0), (1, 1)])
        matching = nx.maximal_matching(G)
        assert len(matching) == 1
        assert not any((u == v for u, v in matching))
        assert nx.is_maximal_matching(G, matching)

    def test_ordering(self):
        """Tests that a maximal matching is computed correctly
        regardless of the order in which nodes are added to the graph.

        """
        for nodes in permutations(range(3)):
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from([(0, 1), (0, 2)])
            matching = nx.maximal_matching(G)
            assert len(matching) == 1
            assert nx.is_maximal_matching(G, matching)

    def test_wrong_graph_type(self):
        error = nx.NetworkXNotImplemented
        raises(error, nx.maximal_matching, nx.MultiGraph())
        raises(error, nx.maximal_matching, nx.MultiDiGraph())
        raises(error, nx.maximal_matching, nx.DiGraph())