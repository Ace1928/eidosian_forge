from itertools import chain, combinations, product
import pytest
import networkx as nx
class TestDAGLCA:

    @classmethod
    def setup_class(cls):
        cls.DG = nx.DiGraph()
        nx.add_path(cls.DG, (0, 1, 2, 3))
        nx.add_path(cls.DG, (0, 4, 3))
        nx.add_path(cls.DG, (0, 5, 6, 8, 3))
        nx.add_path(cls.DG, (5, 7, 8))
        cls.DG.add_edge(6, 2)
        cls.DG.add_edge(7, 2)
        cls.root_distance = nx.shortest_path_length(cls.DG, source=0)
        cls.gold = {(1, 1): 1, (1, 2): 1, (1, 3): 1, (1, 4): 0, (1, 5): 0, (1, 6): 0, (1, 7): 0, (1, 8): 0, (2, 2): 2, (2, 3): 2, (2, 4): 0, (2, 5): 5, (2, 6): 6, (2, 7): 7, (2, 8): 7, (3, 3): 3, (3, 4): 4, (3, 5): 5, (3, 6): 6, (3, 7): 7, (3, 8): 8, (4, 4): 4, (4, 5): 0, (4, 6): 0, (4, 7): 0, (4, 8): 0, (5, 5): 5, (5, 6): 5, (5, 7): 5, (5, 8): 5, (6, 6): 6, (6, 7): 5, (6, 8): 6, (7, 7): 7, (7, 8): 7, (8, 8): 8}
        cls.gold.update((((0, n), 0) for n in cls.DG))

    def assert_lca_dicts_same(self, d1, d2, G=None):
        """Checks if d1 and d2 contain the same pairs and
        have a node at the same distance from root for each.
        If G is None use self.DG."""
        if G is None:
            G = self.DG
            root_distance = self.root_distance
        else:
            roots = [n for n, deg in G.in_degree if deg == 0]
            assert len(roots) == 1
            root_distance = nx.shortest_path_length(G, source=roots[0])
        for a, b in ((min(pair), max(pair)) for pair in chain(d1, d2)):
            assert root_distance[get_pair(d1, a, b)] == root_distance[get_pair(d2, a, b)]

    def test_all_pairs_lca_gold_example(self):
        self.assert_lca_dicts_same(dict(all_pairs_lca(self.DG)), self.gold)

    def test_all_pairs_lca_all_pairs_given(self):
        all_pairs = list(product(self.DG.nodes(), self.DG.nodes()))
        ans = all_pairs_lca(self.DG, pairs=all_pairs)
        self.assert_lca_dicts_same(dict(ans), self.gold)

    def test_all_pairs_lca_generator(self):
        all_pairs = product(self.DG.nodes(), self.DG.nodes())
        ans = all_pairs_lca(self.DG, pairs=all_pairs)
        self.assert_lca_dicts_same(dict(ans), self.gold)

    def test_all_pairs_lca_input_graph_with_two_roots(self):
        G = self.DG.copy()
        G.add_edge(9, 10)
        G.add_edge(9, 4)
        gold = self.gold.copy()
        gold[9, 9] = 9
        gold[9, 10] = 9
        gold[9, 4] = 9
        gold[9, 3] = 9
        gold[10, 4] = 9
        gold[10, 3] = 9
        gold[10, 10] = 10
        testing = dict(all_pairs_lca(G))
        G.add_edge(-1, 9)
        G.add_edge(-1, 0)
        self.assert_lca_dicts_same(testing, gold, G)

    def test_all_pairs_lca_nonexisting_pairs_exception(self):
        pytest.raises(nx.NodeNotFound, all_pairs_lca, self.DG, [(-1, -1)])

    def test_all_pairs_lca_pairs_without_lca(self):
        G = self.DG.copy()
        G.add_node(-1)
        gen = all_pairs_lca(G, [(-1, -1), (-1, 0)])
        assert dict(gen) == {(-1, -1): -1}

    def test_all_pairs_lca_null_graph(self):
        pytest.raises(nx.NetworkXPointlessConcept, all_pairs_lca, nx.DiGraph())

    def test_all_pairs_lca_non_dags(self):
        pytest.raises(nx.NetworkXError, all_pairs_lca, nx.DiGraph([(3, 4), (4, 3)]))

    def test_all_pairs_lca_nonempty_graph_without_lca(self):
        G = nx.DiGraph()
        G.add_node(3)
        ans = list(all_pairs_lca(G))
        assert ans == [((3, 3), 3)]

    def test_all_pairs_lca_bug_gh4942(self):
        G = nx.DiGraph([(0, 2), (1, 2), (2, 3)])
        ans = list(all_pairs_lca(G))
        assert len(ans) == 9

    def test_all_pairs_lca_default_kwarg(self):
        G = nx.DiGraph([(0, 1), (2, 1)])
        sentinel = object()
        assert nx.lowest_common_ancestor(G, 0, 2, default=sentinel) is sentinel

    def test_all_pairs_lca_identity(self):
        G = nx.DiGraph()
        G.add_node(3)
        assert nx.lowest_common_ancestor(G, 3, 3) == 3

    def test_all_pairs_lca_issue_4574(self):
        G = nx.DiGraph()
        G.add_nodes_from(range(17))
        G.add_edges_from([(2, 0), (1, 2), (3, 2), (5, 2), (8, 2), (11, 2), (4, 5), (6, 5), (7, 8), (10, 8), (13, 11), (14, 11), (15, 11), (9, 10), (12, 13), (16, 15)])
        assert nx.lowest_common_ancestor(G, 7, 9) == None

    def test_all_pairs_lca_one_pair_gh4942(self):
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(2, 0)
        G.add_edge(2, 3)
        G.add_edge(4, 0)
        G.add_edge(5, 2)
        assert nx.lowest_common_ancestor(G, 1, 3) == 2