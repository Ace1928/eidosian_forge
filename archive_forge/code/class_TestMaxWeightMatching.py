import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
class TestMaxWeightMatching:
    """Unit tests for the
    :func:`~networkx.algorithms.matching.max_weight_matching` function.

    """

    def test_trivial1(self):
        """Empty graph"""
        G = nx.Graph()
        assert nx.max_weight_matching(G) == set()
        assert nx.min_weight_matching(G) == set()

    def test_selfloop(self):
        G = nx.Graph()
        G.add_edge(0, 0, weight=100)
        assert nx.max_weight_matching(G) == set()
        assert nx.min_weight_matching(G) == set()

    def test_single_edge(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        assert edges_equal(nx.max_weight_matching(G), matching_dict_to_set({0: 1, 1: 0}))
        assert edges_equal(nx.min_weight_matching(G), matching_dict_to_set({0: 1, 1: 0}))

    def test_two_path(self):
        G = nx.Graph()
        G.add_edge('one', 'two', weight=10)
        G.add_edge('two', 'three', weight=11)
        assert edges_equal(nx.max_weight_matching(G), matching_dict_to_set({'three': 'two', 'two': 'three'}))
        assert edges_equal(nx.min_weight_matching(G), matching_dict_to_set({'one': 'two', 'two': 'one'}))

    def test_path(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=5)
        G.add_edge(2, 3, weight=11)
        G.add_edge(3, 4, weight=5)
        assert edges_equal(nx.max_weight_matching(G), matching_dict_to_set({2: 3, 3: 2}))
        assert edges_equal(nx.max_weight_matching(G, 1), matching_dict_to_set({1: 2, 2: 1, 3: 4, 4: 3}))
        assert edges_equal(nx.min_weight_matching(G), matching_dict_to_set({1: 2, 3: 4}))
        assert edges_equal(nx.min_weight_matching(G, 1), matching_dict_to_set({1: 2, 3: 4}))

    def test_square(self):
        G = nx.Graph()
        G.add_edge(1, 4, weight=2)
        G.add_edge(2, 3, weight=2)
        G.add_edge(1, 2, weight=1)
        G.add_edge(3, 4, weight=4)
        assert edges_equal(nx.max_weight_matching(G), matching_dict_to_set({1: 2, 3: 4}))
        assert edges_equal(nx.min_weight_matching(G), matching_dict_to_set({1: 4, 2: 3}))

    def test_edge_attribute_name(self):
        G = nx.Graph()
        G.add_edge('one', 'two', weight=10, abcd=11)
        G.add_edge('two', 'three', weight=11, abcd=10)
        assert edges_equal(nx.max_weight_matching(G, weight='abcd'), matching_dict_to_set({'one': 'two', 'two': 'one'}))
        assert edges_equal(nx.min_weight_matching(G, weight='abcd'), matching_dict_to_set({'three': 'two'}))

    def test_floating_point_weights(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=math.pi)
        G.add_edge(2, 3, weight=math.exp(1))
        G.add_edge(1, 3, weight=3.0)
        G.add_edge(1, 4, weight=math.sqrt(2.0))
        assert edges_equal(nx.max_weight_matching(G), matching_dict_to_set({1: 4, 2: 3, 3: 2, 4: 1}))
        assert edges_equal(nx.min_weight_matching(G), matching_dict_to_set({1: 4, 2: 3, 3: 2, 4: 1}))

    def test_negative_weights(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=2)
        G.add_edge(1, 3, weight=-2)
        G.add_edge(2, 3, weight=1)
        G.add_edge(2, 4, weight=-1)
        G.add_edge(3, 4, weight=-6)
        assert edges_equal(nx.max_weight_matching(G), matching_dict_to_set({1: 2, 2: 1}))
        assert edges_equal(nx.max_weight_matching(G, maxcardinality=True), matching_dict_to_set({1: 3, 2: 4, 3: 1, 4: 2}))
        assert edges_equal(nx.min_weight_matching(G), matching_dict_to_set({1: 2, 3: 4}))

    def test_s_blossom(self):
        """Create S-blossom and use it for augmentation:"""
        G = nx.Graph()
        G.add_weighted_edges_from([(1, 2, 8), (1, 3, 9), (2, 3, 10), (3, 4, 7)])
        answer = matching_dict_to_set({1: 2, 2: 1, 3: 4, 4: 3})
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)
        G.add_weighted_edges_from([(1, 6, 5), (4, 5, 6)])
        answer = matching_dict_to_set({1: 6, 2: 3, 3: 2, 4: 5, 5: 4, 6: 1})
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_s_t_blossom(self):
        """Create S-blossom, relabel as T-blossom, use for augmentation:"""
        G = nx.Graph()
        G.add_weighted_edges_from([(1, 2, 9), (1, 3, 8), (2, 3, 10), (1, 4, 5), (4, 5, 4), (1, 6, 3)])
        answer = matching_dict_to_set({1: 6, 2: 3, 3: 2, 4: 5, 5: 4, 6: 1})
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)
        G.add_edge(4, 5, weight=3)
        G.add_edge(1, 6, weight=4)
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)
        G.remove_edge(1, 6)
        G.add_edge(3, 6, weight=4)
        answer = matching_dict_to_set({1: 2, 2: 1, 3: 6, 4: 5, 5: 4, 6: 3})
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_nested_s_blossom(self):
        """Create nested S-blossom, use for augmentation:"""
        G = nx.Graph()
        G.add_weighted_edges_from([(1, 2, 9), (1, 3, 9), (2, 3, 10), (2, 4, 8), (3, 5, 8), (4, 5, 10), (5, 6, 6)])
        dict_format = {1: 3, 2: 4, 3: 1, 4: 2, 5: 6, 6: 5}
        expected = {frozenset(e) for e in matching_dict_to_set(dict_format)}
        answer = {frozenset(e) for e in nx.max_weight_matching(G)}
        assert answer == expected
        answer = {frozenset(e) for e in nx.min_weight_matching(G)}
        assert answer == expected

    def test_nested_s_blossom_relabel(self):
        """Create S-blossom, relabel as S, include in nested S-blossom:"""
        G = nx.Graph()
        G.add_weighted_edges_from([(1, 2, 10), (1, 7, 10), (2, 3, 12), (3, 4, 20), (3, 5, 20), (4, 5, 25), (5, 6, 10), (6, 7, 10), (7, 8, 8)])
        answer = matching_dict_to_set({1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7})
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_nested_s_blossom_expand(self):
        """Create nested S-blossom, augment, expand recursively:"""
        G = nx.Graph()
        G.add_weighted_edges_from([(1, 2, 8), (1, 3, 8), (2, 3, 10), (2, 4, 12), (3, 5, 12), (4, 5, 14), (4, 6, 12), (5, 7, 12), (6, 7, 14), (7, 8, 12)])
        answer = matching_dict_to_set({1: 2, 2: 1, 3: 5, 4: 6, 5: 3, 6: 4, 7: 8, 8: 7})
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_s_blossom_relabel_expand(self):
        """Create S-blossom, relabel as T, expand:"""
        G = nx.Graph()
        G.add_weighted_edges_from([(1, 2, 23), (1, 5, 22), (1, 6, 15), (2, 3, 25), (3, 4, 22), (4, 5, 25), (4, 8, 14), (5, 7, 13)])
        answer = matching_dict_to_set({1: 6, 2: 3, 3: 2, 4: 8, 5: 7, 6: 1, 7: 5, 8: 4})
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_nested_s_blossom_relabel_expand(self):
        """Create nested S-blossom, relabel as T, expand:"""
        G = nx.Graph()
        G.add_weighted_edges_from([(1, 2, 19), (1, 3, 20), (1, 8, 8), (2, 3, 25), (2, 4, 18), (3, 5, 18), (4, 5, 13), (4, 7, 7), (5, 6, 7)])
        answer = matching_dict_to_set({1: 8, 2: 3, 3: 2, 4: 7, 5: 6, 6: 5, 7: 4, 8: 1})
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_nasty_blossom1(self):
        """Create blossom, relabel as T in more than one way, expand,
        augment:
        """
        G = nx.Graph()
        G.add_weighted_edges_from([(1, 2, 45), (1, 5, 45), (2, 3, 50), (3, 4, 45), (4, 5, 50), (1, 6, 30), (3, 9, 35), (4, 8, 35), (5, 7, 26), (9, 10, 5)])
        ansdict = {1: 6, 2: 3, 3: 2, 4: 8, 5: 7, 6: 1, 7: 5, 8: 4, 9: 10, 10: 9}
        answer = matching_dict_to_set(ansdict)
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_nasty_blossom2(self):
        """Again but slightly different:"""
        G = nx.Graph()
        G.add_weighted_edges_from([(1, 2, 45), (1, 5, 45), (2, 3, 50), (3, 4, 45), (4, 5, 50), (1, 6, 30), (3, 9, 35), (4, 8, 26), (5, 7, 40), (9, 10, 5)])
        ans = {1: 6, 2: 3, 3: 2, 4: 8, 5: 7, 6: 1, 7: 5, 8: 4, 9: 10, 10: 9}
        answer = matching_dict_to_set(ans)
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_nasty_blossom_least_slack(self):
        """Create blossom, relabel as T, expand such that a new
        least-slack S-to-free dge is produced, augment:
        """
        G = nx.Graph()
        G.add_weighted_edges_from([(1, 2, 45), (1, 5, 45), (2, 3, 50), (3, 4, 45), (4, 5, 50), (1, 6, 30), (3, 9, 35), (4, 8, 28), (5, 7, 26), (9, 10, 5)])
        ans = {1: 6, 2: 3, 3: 2, 4: 8, 5: 7, 6: 1, 7: 5, 8: 4, 9: 10, 10: 9}
        answer = matching_dict_to_set(ans)
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_nasty_blossom_augmenting(self):
        """Create nested blossom, relabel as T in more than one way"""
        G = nx.Graph()
        G.add_weighted_edges_from([(1, 2, 45), (1, 7, 45), (2, 3, 50), (3, 4, 45), (4, 5, 95), (4, 6, 94), (5, 6, 94), (6, 7, 50), (1, 8, 30), (3, 11, 35), (5, 9, 36), (7, 10, 26), (11, 12, 5)])
        ans = {1: 8, 2: 3, 3: 2, 4: 6, 5: 9, 6: 4, 7: 10, 8: 1, 9: 5, 10: 7, 11: 12, 12: 11}
        answer = matching_dict_to_set(ans)
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_nasty_blossom_expand_recursively(self):
        """Create nested S-blossom, relabel as S, expand recursively:"""
        G = nx.Graph()
        G.add_weighted_edges_from([(1, 2, 40), (1, 3, 40), (2, 3, 60), (2, 4, 55), (3, 5, 55), (4, 5, 50), (1, 8, 15), (5, 7, 30), (7, 6, 10), (8, 10, 10), (4, 9, 30)])
        ans = {1: 2, 2: 1, 3: 5, 4: 9, 5: 3, 6: 7, 7: 6, 8: 10, 9: 4, 10: 8}
        answer = matching_dict_to_set(ans)
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_wrong_graph_type(self):
        error = nx.NetworkXNotImplemented
        raises(error, nx.max_weight_matching, nx.MultiGraph())
        raises(error, nx.max_weight_matching, nx.MultiDiGraph())
        raises(error, nx.max_weight_matching, nx.DiGraph())
        raises(error, nx.min_weight_matching, nx.DiGraph())