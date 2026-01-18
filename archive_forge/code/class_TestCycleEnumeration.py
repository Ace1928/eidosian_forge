from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
class TestCycleEnumeration:

    @staticmethod
    def K(n):
        return nx.complete_graph(n)

    @staticmethod
    def D(n):
        return nx.complete_graph(n).to_directed()

    @staticmethod
    def edgeset_function(g):
        if g.is_directed():
            return directed_cycle_edgeset
        elif g.is_multigraph():
            return multigraph_cycle_edgeset
        else:
            return undirected_cycle_edgeset

    def check_cycle(self, g, c, es, cache, source, original_c, length_bound, chordless):
        if length_bound is not None and len(c) > length_bound:
            raise RuntimeError(f'computed cycle {original_c} exceeds length bound {length_bound}')
        if source == 'computed':
            if es in cache:
                raise RuntimeError(f'computed cycle {original_c} has already been found!')
            else:
                cache[es] = tuple(original_c)
        elif es in cache:
            cache.pop(es)
        else:
            raise RuntimeError(f'expected cycle {original_c} was not computed')
        if not all((g.has_edge(*e) for e in es)):
            raise RuntimeError(f'{source} claimed cycle {original_c} is not a cycle of g')
        if chordless and len(g.subgraph(c).edges) > len(c):
            raise RuntimeError(f'{source} cycle {original_c} is not chordless')

    def check_cycle_algorithm(self, g, expected_cycles, length_bound=None, chordless=False, algorithm=None):
        if algorithm is None:
            algorithm = nx.chordless_cycles if chordless else nx.simple_cycles
        relabel = list(range(len(g)))
        shuffle(relabel)
        label = dict(zip(g, relabel))
        unlabel = dict(zip(relabel, g))
        h = nx.relabel_nodes(g, label, copy=True)
        edgeset = self.edgeset_function(h)
        params = {}
        if length_bound is not None:
            params['length_bound'] = length_bound
        cycle_cache = {}
        for c in algorithm(h, **params):
            original_c = [unlabel[x] for x in c]
            es = edgeset(c)
            self.check_cycle(h, c, es, cycle_cache, 'computed', original_c, length_bound, chordless)
        if isinstance(expected_cycles, int):
            if len(cycle_cache) != expected_cycles:
                raise RuntimeError(f'expected {expected_cycles} cycles, got {len(cycle_cache)}')
            return
        for original_c in expected_cycles:
            c = [label[x] for x in original_c]
            es = edgeset(c)
            self.check_cycle(h, c, es, cycle_cache, 'expected', original_c, length_bound, chordless)
        if len(cycle_cache):
            for c in cycle_cache.values():
                raise RuntimeError(f'computed cycle {c} is valid but not in the expected cycle set!')

    def check_cycle_enumeration_integer_sequence(self, g_family, cycle_counts, length_bound=None, chordless=False, algorithm=None):
        for g, num_cycles in zip(g_family, cycle_counts):
            self.check_cycle_algorithm(g, num_cycles, length_bound=length_bound, chordless=chordless, algorithm=algorithm)

    def test_directed_chordless_cycle_digons(self):
        g = nx.DiGraph()
        nx.add_cycle(g, range(5))
        nx.add_cycle(g, range(5)[::-1])
        g.add_edge(0, 0)
        expected_cycles = [(0,), (1, 2), (2, 3), (3, 4)]
        self.check_cycle_algorithm(g, expected_cycles, chordless=True)
        self.check_cycle_algorithm(g, expected_cycles, chordless=True, length_bound=2)
        expected_cycles = [c for c in expected_cycles if len(c) < 2]
        self.check_cycle_algorithm(g, expected_cycles, chordless=True, length_bound=1)

    def test_directed_chordless_cycle_undirected(self):
        g = nx.DiGraph([(1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (5, 1), (0, 2)])
        expected_cycles = [(0, 2, 3, 4, 5), (1, 2, 3, 4, 5)]
        self.check_cycle_algorithm(g, expected_cycles, chordless=True)
        g = nx.DiGraph()
        nx.add_cycle(g, range(5))
        nx.add_cycle(g, range(4, 9))
        g.add_edge(7, 3)
        expected_cycles = [(0, 1, 2, 3, 4), (3, 4, 5, 6, 7), (4, 5, 6, 7, 8)]
        self.check_cycle_algorithm(g, expected_cycles, chordless=True)
        g.add_edge(3, 7)
        expected_cycles = [(0, 1, 2, 3, 4), (3, 7), (4, 5, 6, 7, 8)]
        self.check_cycle_algorithm(g, expected_cycles, chordless=True)
        expected_cycles = [(3, 7)]
        self.check_cycle_algorithm(g, expected_cycles, chordless=True, length_bound=4)
        g.remove_edge(7, 3)
        expected_cycles = [(0, 1, 2, 3, 4), (4, 5, 6, 7, 8)]
        self.check_cycle_algorithm(g, expected_cycles, chordless=True)
        g = nx.DiGraph(((i, j) for i in range(10) for j in range(i)))
        expected_cycles = []
        self.check_cycle_algorithm(g, expected_cycles, chordless=True)

    def test_chordless_cycles_directed(self):
        G = nx.DiGraph()
        nx.add_cycle(G, range(5))
        nx.add_cycle(G, range(4, 12))
        expected = [[*range(5)], [*range(4, 12)]]
        self.check_cycle_algorithm(G, expected, chordless=True)
        self.check_cycle_algorithm(G, [c for c in expected if len(c) <= 5], length_bound=5, chordless=True)
        G.add_edge(7, 3)
        expected.append([*range(3, 8)])
        self.check_cycle_algorithm(G, expected, chordless=True)
        self.check_cycle_algorithm(G, [c for c in expected if len(c) <= 5], length_bound=5, chordless=True)
        G.add_edge(3, 7)
        expected[-1] = [7, 3]
        self.check_cycle_algorithm(G, expected, chordless=True)
        self.check_cycle_algorithm(G, [c for c in expected if len(c) <= 5], length_bound=5, chordless=True)
        expected.pop()
        G.remove_edge(7, 3)
        self.check_cycle_algorithm(G, expected, chordless=True)
        self.check_cycle_algorithm(G, [c for c in expected if len(c) <= 5], length_bound=5, chordless=True)

    def test_directed_chordless_cycle_diclique(self):
        g_family = [self.D(n) for n in range(10)]
        expected_cycles = [(n * n - n) // 2 for n in range(10)]
        self.check_cycle_enumeration_integer_sequence(g_family, expected_cycles, chordless=True)
        expected_cycles = [(n * n - n) // 2 for n in range(10)]
        self.check_cycle_enumeration_integer_sequence(g_family, expected_cycles, length_bound=2)

    def test_directed_chordless_loop_blockade(self):
        g = nx.DiGraph(((i, i) for i in range(10)))
        nx.add_cycle(g, range(10))
        expected_cycles = [(i,) for i in range(10)]
        self.check_cycle_algorithm(g, expected_cycles, chordless=True)
        self.check_cycle_algorithm(g, expected_cycles, length_bound=1)
        g = nx.MultiDiGraph(g)
        g.add_edges_from(((i, i) for i in range(0, 10, 2)))
        expected_cycles = [(i,) for i in range(1, 10, 2)]
        self.check_cycle_algorithm(g, expected_cycles, chordless=True)

    def test_simple_cycles_notable_clique_sequences(self):
        g_family = [self.K(n) for n in range(2, 12)]
        expected = [0, 1, 4, 10, 20, 35, 56, 84, 120, 165, 220]
        self.check_cycle_enumeration_integer_sequence(g_family, expected, length_bound=3)

        def triangles(g, **kwargs):
            yield from (c for c in nx.simple_cycles(g, **kwargs) if len(c) == 3)
        g_family = [self.D(n) for n in range(2, 12)]
        expected = [2 * e for e in expected]
        self.check_cycle_enumeration_integer_sequence(g_family, expected, length_bound=3, algorithm=triangles)

        def four_cycles(g, **kwargs):
            yield from (c for c in nx.simple_cycles(g, **kwargs) if len(c) == 4)
        expected = [0, 0, 0, 3, 15, 45, 105, 210, 378, 630, 990]
        g_family = [self.K(n) for n in range(1, 12)]
        self.check_cycle_enumeration_integer_sequence(g_family, expected, length_bound=4, algorithm=four_cycles)
        expected = [2 * e for e in expected]
        g_family = [self.D(n) for n in range(1, 15)]
        self.check_cycle_enumeration_integer_sequence(g_family, expected, length_bound=4, algorithm=four_cycles)
        expected = [0, 1, 5, 20, 84, 409, 2365]
        g_family = [self.D(n) for n in range(1, 8)]
        self.check_cycle_enumeration_integer_sequence(g_family, expected)
        expected = [0, 0, 0, 1, 7, 37, 197, 1172]
        g_family = [self.K(n) for n in range(8)]
        self.check_cycle_enumeration_integer_sequence(g_family, expected)

    def test_directed_chordless_cycle_parallel_multiedges(self):
        g = nx.MultiGraph()
        nx.add_cycle(g, range(5))
        expected = [[*range(5)]]
        self.check_cycle_algorithm(g, expected, chordless=True)
        nx.add_cycle(g, range(5))
        expected = [*cycle_edges(range(5))]
        self.check_cycle_algorithm(g, expected, chordless=True)
        nx.add_cycle(g, range(5))
        expected = []
        self.check_cycle_algorithm(g, expected, chordless=True)
        g = nx.MultiDiGraph()
        nx.add_cycle(g, range(5))
        expected = [[*range(5)]]
        self.check_cycle_algorithm(g, expected, chordless=True)
        nx.add_cycle(g, range(5))
        self.check_cycle_algorithm(g, [], chordless=True)
        nx.add_cycle(g, range(5))
        self.check_cycle_algorithm(g, [], chordless=True)
        g = nx.MultiDiGraph()
        nx.add_cycle(g, range(5))
        nx.add_cycle(g, range(5)[::-1])
        expected = [*cycle_edges(range(5))]
        self.check_cycle_algorithm(g, expected, chordless=True)
        nx.add_cycle(g, range(5))
        self.check_cycle_algorithm(g, [], chordless=True)

    def test_chordless_cycles_graph(self):
        G = nx.Graph()
        nx.add_cycle(G, range(5))
        nx.add_cycle(G, range(4, 12))
        expected = [[*range(5)], [*range(4, 12)]]
        self.check_cycle_algorithm(G, expected, chordless=True)
        self.check_cycle_algorithm(G, [c for c in expected if len(c) <= 5], length_bound=5, chordless=True)
        G.add_edge(7, 3)
        expected.append([*range(3, 8)])
        expected.append([4, 3, 7, 8, 9, 10, 11])
        self.check_cycle_algorithm(G, expected, chordless=True)
        self.check_cycle_algorithm(G, [c for c in expected if len(c) <= 5], length_bound=5, chordless=True)

    def test_chordless_cycles_giant_hamiltonian(self):
        n = 1000
        assert n % 2 == 0
        G = nx.Graph()
        for v in range(n):
            if not v % 2:
                G.add_edge(v, (v + 2) % n)
            G.add_edge(v, (v + 1) % n)
        expected = [[*range(0, n, 2)]] + [[x % n for x in range(i, i + 3)] for i in range(0, n, 2)]
        self.check_cycle_algorithm(G, expected, chordless=True)
        self.check_cycle_algorithm(G, [c for c in expected if len(c) <= 3], length_bound=3, chordless=True)
        n = 100
        assert n % 2 == 0
        G = nx.DiGraph()
        for v in range(n):
            G.add_edge(v, (v + 1) % n)
            if not v % 2:
                G.add_edge((v + 2) % n, v)
        expected = [[*range(n - 2, -2, -2)]] + [[x % n for x in range(i, i + 3)] for i in range(0, n, 2)]
        self.check_cycle_algorithm(G, expected, chordless=True)
        self.check_cycle_algorithm(G, [c for c in expected if len(c) <= 3], length_bound=3, chordless=True)

    def test_simple_cycles_acyclic_tournament(self):
        n = 10
        G = nx.DiGraph(((x, y) for x in range(n) for y in range(x)))
        self.check_cycle_algorithm(G, [])
        self.check_cycle_algorithm(G, [], chordless=True)
        for k in range(n + 1):
            self.check_cycle_algorithm(G, [], length_bound=k)
            self.check_cycle_algorithm(G, [], length_bound=k, chordless=True)

    def test_simple_cycles_graph(self):
        testG = nx.cycle_graph(8)
        cyc1 = tuple(range(8))
        self.check_cycle_algorithm(testG, [cyc1])
        testG.add_edge(4, -1)
        nx.add_path(testG, [3, -2, -3, -4])
        self.check_cycle_algorithm(testG, [cyc1])
        testG.update(nx.cycle_graph(range(8, 16)))
        cyc2 = tuple(range(8, 16))
        self.check_cycle_algorithm(testG, [cyc1, cyc2])
        testG.update(nx.cycle_graph(range(4, 12)))
        cyc3 = tuple(range(4, 12))
        expected = {(0, 1, 2, 3, 4, 5, 6, 7), (8, 9, 10, 11, 12, 13, 14, 15), (4, 5, 6, 7, 8, 9, 10, 11), (4, 5, 6, 7, 8, 15, 14, 13, 12, 11), (0, 1, 2, 3, 4, 11, 10, 9, 8, 7), (0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 8, 7)}
        self.check_cycle_algorithm(testG, expected)
        assert len(expected) == 2 ** 3 - 1 - 1
        testG = nx.cycle_graph(12)
        testG.update(nx.cycle_graph([12, 10, 13, 2, 14, 4, 15, 8]).edges)
        expected = 2 ** 5 - 1 - 11
        self.check_cycle_algorithm(testG, expected)

    def test_simple_cycles_bounded(self):
        d = nx.DiGraph()
        expected = []
        for n in range(10):
            nx.add_cycle(d, range(n))
            expected.append(n)
            for k, e in enumerate(expected):
                self.check_cycle_algorithm(d, e, length_bound=k)
        g = nx.Graph()
        top = 0
        expected = []
        for n in range(10):
            expected.append(n if n < 2 else n - 1)
            if n == 2:
                continue
            nx.add_cycle(g, range(top, top + n))
            top += n
            for k, e in enumerate(expected):
                self.check_cycle_algorithm(g, e, length_bound=k)

    def test_simple_cycles_bound_corner_cases(self):
        G = nx.cycle_graph(4)
        DG = nx.cycle_graph(4, create_using=nx.DiGraph)
        assert list(nx.simple_cycles(G, length_bound=0)) == []
        assert list(nx.simple_cycles(DG, length_bound=0)) == []
        assert list(nx.chordless_cycles(G, length_bound=0)) == []
        assert list(nx.chordless_cycles(DG, length_bound=0)) == []

    def test_simple_cycles_bound_error(self):
        with pytest.raises(ValueError):
            G = nx.DiGraph()
            for c in nx.simple_cycles(G, -1):
                assert False
        with pytest.raises(ValueError):
            G = nx.Graph()
            for c in nx.simple_cycles(G, -1):
                assert False
        with pytest.raises(ValueError):
            G = nx.Graph()
            for c in nx.chordless_cycles(G, -1):
                assert False
        with pytest.raises(ValueError):
            G = nx.DiGraph()
            for c in nx.chordless_cycles(G, -1):
                assert False

    def test_chordless_cycles_clique(self):
        g_family = [self.K(n) for n in range(2, 15)]
        expected = [0, 1, 4, 10, 20, 35, 56, 84, 120, 165, 220, 286, 364]
        self.check_cycle_enumeration_integer_sequence(g_family, expected, chordless=True)
        expected = [(n * n - n) // 2 for n in range(15)]
        g_family = [self.D(n) for n in range(15)]
        self.check_cycle_enumeration_integer_sequence(g_family, expected, chordless=True)