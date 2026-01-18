from itertools import combinations
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal
class TestEdgeBoundary:
    """Unit tests for the :func:`~networkx.edge_boundary` function."""

    def test_null_graph(self):
        null = nx.null_graph()
        assert list(nx.edge_boundary(null, [])) == []
        assert list(nx.edge_boundary(null, [], [])) == []
        assert list(nx.edge_boundary(null, [1, 2, 3])) == []
        assert list(nx.edge_boundary(null, [1, 2, 3], [4, 5, 6])) == []
        assert list(nx.edge_boundary(null, [1, 2, 3], [3, 4, 5])) == []

    def test_path_graph(self):
        P10 = cnlti(nx.path_graph(10), first_label=1)
        assert list(nx.edge_boundary(P10, [])) == []
        assert list(nx.edge_boundary(P10, [], [])) == []
        assert list(nx.edge_boundary(P10, [1, 2, 3])) == [(3, 4)]
        assert sorted(nx.edge_boundary(P10, [4, 5, 6])) == [(4, 3), (6, 7)]
        assert sorted(nx.edge_boundary(P10, [3, 4, 5, 6, 7])) == [(3, 2), (7, 8)]
        assert list(nx.edge_boundary(P10, [8, 9, 10])) == [(8, 7)]
        assert sorted(nx.edge_boundary(P10, [4, 5, 6], [9, 10])) == []
        assert list(nx.edge_boundary(P10, [1, 2, 3], [3, 4, 5])) == [(2, 3), (3, 4)]

    def test_complete_graph(self):
        K10 = cnlti(nx.complete_graph(10), first_label=1)

        def ilen(iterable):
            return sum((1 for i in iterable))
        assert list(nx.edge_boundary(K10, [])) == []
        assert list(nx.edge_boundary(K10, [], [])) == []
        assert ilen(nx.edge_boundary(K10, [1, 2, 3])) == 21
        assert ilen(nx.edge_boundary(K10, [4, 5, 6, 7])) == 24
        assert ilen(nx.edge_boundary(K10, [3, 4, 5, 6, 7])) == 25
        assert ilen(nx.edge_boundary(K10, [8, 9, 10])) == 21
        assert edges_equal(nx.edge_boundary(K10, [4, 5, 6], [9, 10]), [(4, 9), (4, 10), (5, 9), (5, 10), (6, 9), (6, 10)])
        assert edges_equal(nx.edge_boundary(K10, [1, 2, 3], [3, 4, 5]), [(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5)])

    def test_directed(self):
        """Tests the edge boundary of a directed graph."""
        G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
        S = {0, 1}
        boundary = list(nx.edge_boundary(G, S))
        expected = [(1, 2)]
        assert boundary == expected

    def test_multigraph(self):
        """Tests the edge boundary of a multigraph."""
        G = nx.MultiGraph(list(nx.cycle_graph(5).edges()) * 2)
        S = {0, 1}
        boundary = list(nx.edge_boundary(G, S))
        expected = [(0, 4), (0, 4), (1, 2), (1, 2)]
        assert boundary == expected

    def test_multidigraph(self):
        """Tests the edge boundary of a multidigraph."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        G = nx.MultiDiGraph(edges * 2)
        S = {0, 1}
        boundary = list(nx.edge_boundary(G, S))
        expected = [(1, 2), (1, 2)]
        assert boundary == expected