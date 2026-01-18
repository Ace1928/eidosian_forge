from itertools import product
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
class TestNestedTuple:
    """Unit tests for the nested tuple encoding and decoding functions."""

    def test_nontree(self):
        with pytest.raises(nx.NotATree):
            G = nx.cycle_graph(3)
            nx.to_nested_tuple(G, 0)

    def test_unknown_root(self):
        with pytest.raises(nx.NodeNotFound):
            G = nx.path_graph(2)
            nx.to_nested_tuple(G, 'bogus')

    def test_encoding(self):
        T = nx.full_rary_tree(2, 2 ** 3 - 1)
        expected = (((), ()), ((), ()))
        actual = nx.to_nested_tuple(T, 0)
        assert nodes_equal(expected, actual)

    def test_canonical_form(self):
        T = nx.Graph()
        T.add_edges_from([(0, 1), (0, 2), (0, 3)])
        T.add_edges_from([(1, 4), (1, 5)])
        T.add_edges_from([(3, 6), (3, 7)])
        root = 0
        actual = nx.to_nested_tuple(T, root, canonical_form=True)
        expected = ((), ((), ()), ((), ()))
        assert actual == expected

    def test_decoding(self):
        balanced = (((), ()), ((), ()))
        expected = nx.full_rary_tree(2, 2 ** 3 - 1)
        actual = nx.from_nested_tuple(balanced)
        assert nx.is_isomorphic(expected, actual)

    def test_sensible_relabeling(self):
        balanced = (((), ()), ((), ()))
        T = nx.from_nested_tuple(balanced, sensible_relabeling=True)
        edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
        assert nodes_equal(list(T), list(range(2 ** 3 - 1)))
        assert edges_equal(list(T.edges()), edges)