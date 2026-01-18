import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
class TestGeneratorLine:

    def test_star(self):
        G = nx.star_graph(5)
        L = nx.line_graph(G)
        assert nx.is_isomorphic(L, nx.complete_graph(5))

    def test_path(self):
        G = nx.path_graph(5)
        L = nx.line_graph(G)
        assert nx.is_isomorphic(L, nx.path_graph(4))

    def test_cycle(self):
        G = nx.cycle_graph(5)
        L = nx.line_graph(G)
        assert nx.is_isomorphic(L, G)

    def test_digraph1(self):
        G = nx.DiGraph([(0, 1), (0, 2), (0, 3)])
        L = nx.line_graph(G)
        assert L.adj == {(0, 1): {}, (0, 2): {}, (0, 3): {}}

    def test_multigraph1(self):
        G = nx.MultiGraph([(0, 1), (0, 1), (1, 0), (0, 2), (2, 0), (0, 3)])
        L = nx.line_graph(G)
        assert edges_equal(L.edges(), [((0, 3, 0), (0, 1, 0)), ((0, 3, 0), (0, 2, 0)), ((0, 3, 0), (0, 2, 1)), ((0, 3, 0), (0, 1, 1)), ((0, 3, 0), (0, 1, 2)), ((0, 1, 0), (0, 1, 1)), ((0, 1, 0), (0, 2, 0)), ((0, 1, 0), (0, 1, 2)), ((0, 1, 0), (0, 2, 1)), ((0, 1, 1), (0, 1, 2)), ((0, 1, 1), (0, 2, 0)), ((0, 1, 1), (0, 2, 1)), ((0, 1, 2), (0, 2, 0)), ((0, 1, 2), (0, 2, 1)), ((0, 2, 0), (0, 2, 1))])

    def test_multigraph2(self):
        G = nx.MultiGraph([(1, 2), (2, 1)])
        L = nx.line_graph(G)
        assert edges_equal(L.edges(), [((1, 2, 0), (1, 2, 1))])

    def test_multidigraph1(self):
        G = nx.MultiDiGraph([(1, 2), (2, 1)])
        L = nx.line_graph(G)
        assert edges_equal(L.edges(), [((1, 2, 0), (2, 1, 0)), ((2, 1, 0), (1, 2, 0))])

    def test_multidigraph2(self):
        G = nx.MultiDiGraph([(0, 1), (0, 1), (0, 1), (1, 2)])
        L = nx.line_graph(G)
        assert edges_equal(L.edges(), [((0, 1, 0), (1, 2, 0)), ((0, 1, 1), (1, 2, 0)), ((0, 1, 2), (1, 2, 0))])

    def test_digraph2(self):
        G = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
        L = nx.line_graph(G)
        assert edges_equal(L.edges(), [((0, 1), (1, 2)), ((1, 2), (2, 3))])

    def test_create1(self):
        G = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
        L = nx.line_graph(G, create_using=nx.Graph())
        assert edges_equal(L.edges(), [((0, 1), (1, 2)), ((1, 2), (2, 3))])

    def test_create2(self):
        G = nx.Graph([(0, 1), (1, 2), (2, 3)])
        L = nx.line_graph(G, create_using=nx.DiGraph())
        assert edges_equal(L.edges(), [((0, 1), (1, 2)), ((1, 2), (2, 3))])