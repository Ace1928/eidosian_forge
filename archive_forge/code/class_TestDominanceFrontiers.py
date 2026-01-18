import pytest
import networkx as nx
class TestDominanceFrontiers:

    def test_exceptions(self):
        G = nx.Graph()
        G.add_node(0)
        pytest.raises(nx.NetworkXNotImplemented, nx.dominance_frontiers, G, 0)
        G = nx.MultiGraph(G)
        pytest.raises(nx.NetworkXNotImplemented, nx.dominance_frontiers, G, 0)
        G = nx.DiGraph([[0, 0]])
        pytest.raises(nx.NetworkXError, nx.dominance_frontiers, G, 1)

    def test_singleton(self):
        G = nx.DiGraph()
        G.add_node(0)
        assert nx.dominance_frontiers(G, 0) == {0: set()}
        G.add_edge(0, 0)
        assert nx.dominance_frontiers(G, 0) == {0: set()}

    def test_path(self):
        n = 5
        G = nx.path_graph(n, create_using=nx.DiGraph())
        assert nx.dominance_frontiers(G, 0) == {i: set() for i in range(n)}

    def test_cycle(self):
        n = 5
        G = nx.cycle_graph(n, create_using=nx.DiGraph())
        assert nx.dominance_frontiers(G, 0) == {i: set() for i in range(n)}

    def test_unreachable(self):
        n = 5
        assert n > 1
        G = nx.path_graph(n, create_using=nx.DiGraph())
        assert nx.dominance_frontiers(G, n // 2) == {i: set() for i in range(n // 2, n)}

    def test_irreducible1(self):
        edges = [(1, 2), (2, 1), (3, 2), (4, 1), (5, 3), (5, 4)]
        G = nx.DiGraph(edges)
        assert dict(nx.dominance_frontiers(G, 5).items()) == {1: {2}, 2: {1}, 3: {2}, 4: {1}, 5: set()}

    def test_irreducible2(self):
        edges = [(1, 2), (2, 1), (2, 3), (3, 2), (4, 2), (4, 3), (5, 1), (6, 4), (6, 5)]
        G = nx.DiGraph(edges)
        assert nx.dominance_frontiers(G, 6) == {1: {2}, 2: {1, 3}, 3: {2}, 4: {2, 3}, 5: {1}, 6: set()}

    def test_domrel_png(self):
        edges = [(1, 2), (2, 3), (2, 4), (2, 6), (3, 5), (4, 5), (5, 2)]
        G = nx.DiGraph(edges)
        assert nx.dominance_frontiers(G, 1) == {1: set(), 2: {2}, 3: {5}, 4: {5}, 5: {2}, 6: set()}
        result = nx.dominance_frontiers(G.reverse(copy=False), 6)
        assert result == {1: set(), 2: {2}, 3: {2}, 4: {2}, 5: {2}, 6: set()}

    def test_boost_example(self):
        edges = [(0, 1), (1, 2), (1, 3), (2, 7), (3, 4), (4, 5), (4, 6), (5, 7), (6, 4)]
        G = nx.DiGraph(edges)
        assert nx.dominance_frontiers(G, 0) == {0: set(), 1: set(), 2: {7}, 3: {7}, 4: {4, 7}, 5: {7}, 6: {4}, 7: set()}
        result = nx.dominance_frontiers(G.reverse(copy=False), 7)
        expected = {0: set(), 1: set(), 2: {1}, 3: {1}, 4: {1, 4}, 5: {1}, 6: {4}, 7: set()}
        assert result == expected

    def test_discard_issue(self):
        g = nx.DiGraph()
        g.add_edges_from([('b0', 'b1'), ('b1', 'b2'), ('b2', 'b3'), ('b3', 'b1'), ('b1', 'b5'), ('b5', 'b6'), ('b5', 'b8'), ('b6', 'b7'), ('b8', 'b7'), ('b7', 'b3'), ('b3', 'b4')])
        df = nx.dominance_frontiers(g, 'b0')
        assert df == {'b4': set(), 'b5': {'b3'}, 'b6': {'b7'}, 'b7': {'b3'}, 'b0': set(), 'b1': {'b1'}, 'b2': {'b3'}, 'b3': {'b1'}, 'b8': {'b7'}}

    def test_loop(self):
        g = nx.DiGraph()
        g.add_edges_from([('a', 'b'), ('b', 'c'), ('b', 'a')])
        df = nx.dominance_frontiers(g, 'a')
        assert df == {'a': set(), 'b': set(), 'c': set()}

    def test_missing_immediate_doms(self):
        g = nx.DiGraph()
        edges = [('entry_1', 'b1'), ('b1', 'b2'), ('b2', 'b3'), ('b3', 'exit'), ('entry_2', 'b3')]
        g.add_edges_from(edges)
        nx.dominance_frontiers(g, 'entry_1')

    def test_loops_larger(self):
        g = nx.DiGraph()
        edges = [('entry', 'exit'), ('entry', '1'), ('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('5', '6'), ('6', 'exit'), ('6', '2'), ('5', '3'), ('4', '4')]
        g.add_edges_from(edges)
        df = nx.dominance_frontiers(g, 'entry')
        answer = {'entry': set(), '1': {'exit'}, '2': {'exit', '2'}, '3': {'exit', '3', '2'}, '4': {'exit', '4', '3', '2'}, '5': {'exit', '3', '2'}, '6': {'exit', '2'}, 'exit': set()}
        for n in df:
            assert set(df[n]) == set(answer[n])