from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
class TestDAG:

    @classmethod
    def setup_class(cls):
        pass

    def test_topological_sort1(self):
        DG = nx.DiGraph([(1, 2), (1, 3), (2, 3)])
        for algorithm in [nx.topological_sort, nx.lexicographical_topological_sort]:
            assert tuple(algorithm(DG)) == (1, 2, 3)
        DG.add_edge(3, 2)
        for algorithm in [nx.topological_sort, nx.lexicographical_topological_sort]:
            pytest.raises(nx.NetworkXUnfeasible, _consume, algorithm(DG))
        DG.remove_edge(2, 3)
        for algorithm in [nx.topological_sort, nx.lexicographical_topological_sort]:
            assert tuple(algorithm(DG)) == (1, 3, 2)
        DG.remove_edge(3, 2)
        assert tuple(nx.topological_sort(DG)) in {(1, 2, 3), (1, 3, 2)}
        assert tuple(nx.lexicographical_topological_sort(DG)) == (1, 2, 3)

    def test_is_directed_acyclic_graph(self):
        G = nx.generators.complete_graph(2)
        assert not nx.is_directed_acyclic_graph(G)
        assert not nx.is_directed_acyclic_graph(G.to_directed())
        assert not nx.is_directed_acyclic_graph(nx.Graph([(3, 4), (4, 5)]))
        assert nx.is_directed_acyclic_graph(nx.DiGraph([(3, 4), (4, 5)]))

    def test_topological_sort2(self):
        DG = nx.DiGraph({1: [2], 2: [3], 3: [4], 4: [5], 5: [1], 11: [12], 12: [13], 13: [14], 14: [15]})
        pytest.raises(nx.NetworkXUnfeasible, _consume, nx.topological_sort(DG))
        assert not nx.is_directed_acyclic_graph(DG)
        DG.remove_edge(1, 2)
        _consume(nx.topological_sort(DG))
        assert nx.is_directed_acyclic_graph(DG)

    def test_topological_sort3(self):
        DG = nx.DiGraph()
        DG.add_edges_from([(1, i) for i in range(2, 5)])
        DG.add_edges_from([(2, i) for i in range(5, 9)])
        DG.add_edges_from([(6, i) for i in range(9, 12)])
        DG.add_edges_from([(4, i) for i in range(12, 15)])

        def validate(order):
            assert isinstance(order, list)
            assert set(order) == set(DG)
            for u, v in combinations(order, 2):
                assert not nx.has_path(DG, v, u)
        validate(list(nx.topological_sort(DG)))
        DG.add_edge(14, 1)
        pytest.raises(nx.NetworkXUnfeasible, _consume, nx.topological_sort(DG))

    def test_topological_sort4(self):
        G = nx.Graph()
        G.add_edge(1, 2)
        pytest.raises(nx.NetworkXError, _consume, nx.topological_sort(G))

    def test_topological_sort5(self):
        G = nx.DiGraph()
        G.add_edge(0, 1)
        assert list(nx.topological_sort(G)) == [0, 1]

    def test_topological_sort6(self):
        for algorithm in [nx.topological_sort, nx.lexicographical_topological_sort]:

            def runtime_error():
                DG = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
                first = True
                for x in algorithm(DG):
                    if first:
                        first = False
                        DG.add_edge(5 - x, 5)

            def unfeasible_error():
                DG = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
                first = True
                for x in algorithm(DG):
                    if first:
                        first = False
                        DG.remove_node(4)

            def runtime_error2():
                DG = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
                first = True
                for x in algorithm(DG):
                    if first:
                        first = False
                        DG.remove_node(2)
            pytest.raises(RuntimeError, runtime_error)
            pytest.raises(RuntimeError, runtime_error2)
            pytest.raises(nx.NetworkXUnfeasible, unfeasible_error)

    def test_all_topological_sorts_1(self):
        DG = nx.DiGraph([(1, 2), (2, 3), (3, 4), (4, 5)])
        assert list(nx.all_topological_sorts(DG)) == [[1, 2, 3, 4, 5]]

    def test_all_topological_sorts_2(self):
        DG = nx.DiGraph([(1, 3), (2, 1), (2, 4), (4, 3), (4, 5)])
        assert sorted(nx.all_topological_sorts(DG)) == [[2, 1, 4, 3, 5], [2, 1, 4, 5, 3], [2, 4, 1, 3, 5], [2, 4, 1, 5, 3], [2, 4, 5, 1, 3]]

    def test_all_topological_sorts_3(self):

        def unfeasible():
            DG = nx.DiGraph([(1, 2), (2, 3), (3, 4), (4, 2), (4, 5)])
            list(nx.all_topological_sorts(DG))

        def not_implemented():
            G = nx.Graph([(1, 2), (2, 3)])
            list(nx.all_topological_sorts(G))

        def not_implemented_2():
            G = nx.MultiGraph([(1, 2), (1, 2), (2, 3)])
            list(nx.all_topological_sorts(G))
        pytest.raises(nx.NetworkXUnfeasible, unfeasible)
        pytest.raises(nx.NetworkXNotImplemented, not_implemented)
        pytest.raises(nx.NetworkXNotImplemented, not_implemented_2)

    def test_all_topological_sorts_4(self):
        DG = nx.DiGraph()
        for i in range(7):
            DG.add_node(i)
        assert sorted(map(list, permutations(DG.nodes))) == sorted(nx.all_topological_sorts(DG))

    def test_all_topological_sorts_multigraph_1(self):
        DG = nx.MultiDiGraph([(1, 2), (1, 2), (2, 3), (3, 4), (3, 5), (3, 5), (3, 5)])
        assert sorted(nx.all_topological_sorts(DG)) == sorted([[1, 2, 3, 4, 5], [1, 2, 3, 5, 4]])

    def test_all_topological_sorts_multigraph_2(self):
        N = 9
        edges = []
        for i in range(1, N):
            edges.extend([(i, i + 1)] * i)
        DG = nx.MultiDiGraph(edges)
        assert list(nx.all_topological_sorts(DG)) == [list(range(1, N + 1))]

    def test_ancestors(self):
        G = nx.DiGraph()
        ancestors = nx.algorithms.dag.ancestors
        G.add_edges_from([(1, 2), (1, 3), (4, 2), (4, 3), (4, 5), (2, 6), (5, 6)])
        assert ancestors(G, 6) == {1, 2, 4, 5}
        assert ancestors(G, 3) == {1, 4}
        assert ancestors(G, 1) == set()
        pytest.raises(nx.NetworkXError, ancestors, G, 8)

    def test_descendants(self):
        G = nx.DiGraph()
        descendants = nx.algorithms.dag.descendants
        G.add_edges_from([(1, 2), (1, 3), (4, 2), (4, 3), (4, 5), (2, 6), (5, 6)])
        assert descendants(G, 1) == {2, 3, 6}
        assert descendants(G, 4) == {2, 3, 5, 6}
        assert descendants(G, 3) == set()
        pytest.raises(nx.NetworkXError, descendants, G, 8)

    def test_transitive_closure(self):
        G = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        assert edges_equal(nx.transitive_closure(G).edges(), solution)
        G = nx.DiGraph([(1, 2), (2, 3), (2, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
        assert edges_equal(nx.transitive_closure(G).edges(), solution)
        G = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
        solution = [(1, 2), (2, 1), (2, 3), (3, 2), (1, 3), (3, 1)]
        soln = sorted(solution + [(n, n) for n in G])
        assert edges_equal(sorted(nx.transitive_closure(G).edges()), soln)
        G = nx.Graph([(1, 2), (2, 3), (3, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        assert edges_equal(sorted(nx.transitive_closure(G).edges()), solution)
        G = nx.MultiGraph([(1, 2), (2, 3), (3, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        assert edges_equal(sorted(nx.transitive_closure(G).edges()), solution)
        G = nx.MultiDiGraph([(1, 2), (2, 3), (3, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        assert edges_equal(sorted(nx.transitive_closure(G).edges()), solution)
        G = nx.DiGraph([(1, 2, {'a': 3}), (2, 3, {'b': 0}), (3, 4)])
        H = nx.transitive_closure(G)
        for u, v in G.edges():
            assert G.get_edge_data(u, v) == H.get_edge_data(u, v)
        k = 10
        G = nx.DiGraph(((i, i + 1, {'f': 'b', 'weight': i}) for i in range(k)))
        H = nx.transitive_closure(G)
        for u, v in G.edges():
            assert G.get_edge_data(u, v) == H.get_edge_data(u, v)
        G = nx.Graph()
        with pytest.raises(nx.NetworkXError):
            nx.transitive_closure(G, reflexive='wrong input')

    def test_reflexive_transitive_closure(self):
        G = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        soln = sorted(solution + [(n, n) for n in G])
        assert edges_equal(nx.transitive_closure(G).edges(), solution)
        assert edges_equal(nx.transitive_closure(G, False).edges(), solution)
        assert edges_equal(nx.transitive_closure(G, True).edges(), soln)
        assert edges_equal(nx.transitive_closure(G, None).edges(), solution)
        G = nx.DiGraph([(1, 2), (2, 3), (2, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
        soln = sorted(solution + [(n, n) for n in G])
        assert edges_equal(nx.transitive_closure(G).edges(), solution)
        assert edges_equal(nx.transitive_closure(G, False).edges(), solution)
        assert edges_equal(nx.transitive_closure(G, True).edges(), soln)
        assert edges_equal(nx.transitive_closure(G, None).edges(), solution)
        G = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
        solution = sorted([(1, 2), (2, 1), (2, 3), (3, 2), (1, 3), (3, 1)])
        soln = sorted(solution + [(n, n) for n in G])
        assert edges_equal(sorted(nx.transitive_closure(G).edges()), soln)
        assert edges_equal(sorted(nx.transitive_closure(G, False).edges()), soln)
        assert edges_equal(sorted(nx.transitive_closure(G, None).edges()), solution)
        assert edges_equal(sorted(nx.transitive_closure(G, True).edges()), soln)
        G = nx.Graph([(1, 2), (2, 3), (3, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        soln = sorted(solution + [(n, n) for n in G])
        assert edges_equal(nx.transitive_closure(G).edges(), solution)
        assert edges_equal(nx.transitive_closure(G, False).edges(), solution)
        assert edges_equal(nx.transitive_closure(G, True).edges(), soln)
        assert edges_equal(nx.transitive_closure(G, None).edges(), solution)
        G = nx.MultiGraph([(1, 2), (2, 3), (3, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        soln = sorted(solution + [(n, n) for n in G])
        assert edges_equal(nx.transitive_closure(G).edges(), solution)
        assert edges_equal(nx.transitive_closure(G, False).edges(), solution)
        assert edges_equal(nx.transitive_closure(G, True).edges(), soln)
        assert edges_equal(nx.transitive_closure(G, None).edges(), solution)
        G = nx.MultiDiGraph([(1, 2), (2, 3), (3, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        soln = sorted(solution + [(n, n) for n in G])
        assert edges_equal(nx.transitive_closure(G).edges(), solution)
        assert edges_equal(nx.transitive_closure(G, False).edges(), solution)
        assert edges_equal(nx.transitive_closure(G, True).edges(), soln)
        assert edges_equal(nx.transitive_closure(G, None).edges(), solution)

    def test_transitive_closure_dag(self):
        G = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
        transitive_closure = nx.algorithms.dag.transitive_closure_dag
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        assert edges_equal(transitive_closure(G).edges(), solution)
        G = nx.DiGraph([(1, 2), (2, 3), (2, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
        assert edges_equal(transitive_closure(G).edges(), solution)
        G = nx.Graph([(1, 2), (2, 3), (3, 4)])
        pytest.raises(nx.NetworkXNotImplemented, transitive_closure, G)
        G = nx.DiGraph([(1, 2, {'a': 3}), (2, 3, {'b': 0}), (3, 4)])
        H = transitive_closure(G)
        for u, v in G.edges():
            assert G.get_edge_data(u, v) == H.get_edge_data(u, v)
        k = 10
        G = nx.DiGraph(((i, i + 1, {'foo': 'bar', 'weight': i}) for i in range(k)))
        H = transitive_closure(G)
        for u, v in G.edges():
            assert G.get_edge_data(u, v) == H.get_edge_data(u, v)

    def test_transitive_reduction(self):
        G = nx.DiGraph([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])
        transitive_reduction = nx.algorithms.dag.transitive_reduction
        solution = [(1, 2), (2, 3), (3, 4)]
        assert edges_equal(transitive_reduction(G).edges(), solution)
        G = nx.DiGraph([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)])
        transitive_reduction = nx.algorithms.dag.transitive_reduction
        solution = [(1, 2), (2, 3), (2, 4)]
        assert edges_equal(transitive_reduction(G).edges(), solution)
        G = nx.Graph([(1, 2), (2, 3), (3, 4)])
        pytest.raises(nx.NetworkXNotImplemented, transitive_reduction, G)

    def _check_antichains(self, solution, result):
        sol = [frozenset(a) for a in solution]
        res = [frozenset(a) for a in result]
        assert set(sol) == set(res)

    def test_antichains(self):
        antichains = nx.algorithms.dag.antichains
        G = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
        solution = [[], [4], [3], [2], [1]]
        self._check_antichains(list(antichains(G)), solution)
        G = nx.DiGraph([(1, 2), (2, 3), (2, 4), (3, 5), (5, 6), (5, 7)])
        solution = [[], [4], [7], [7, 4], [6], [6, 4], [6, 7], [6, 7, 4], [5], [5, 4], [3], [3, 4], [2], [1]]
        self._check_antichains(list(antichains(G)), solution)
        G = nx.DiGraph([(1, 2), (1, 3), (3, 4), (3, 5), (5, 6)])
        solution = [[], [6], [5], [4], [4, 6], [4, 5], [3], [2], [2, 6], [2, 5], [2, 4], [2, 4, 6], [2, 4, 5], [2, 3], [1]]
        self._check_antichains(list(antichains(G)), solution)
        G = nx.DiGraph({0: [1, 2], 1: [4], 2: [3], 3: [4]})
        solution = [[], [4], [3], [2], [1], [1, 3], [1, 2], [0]]
        self._check_antichains(list(antichains(G)), solution)
        G = nx.DiGraph()
        self._check_antichains(list(antichains(G)), [[]])
        G = nx.DiGraph()
        G.add_nodes_from([0, 1, 2])
        solution = [[], [0], [1], [1, 0], [2], [2, 0], [2, 1], [2, 1, 0]]
        self._check_antichains(list(antichains(G)), solution)

        def f(x):
            return list(antichains(x))
        G = nx.Graph([(1, 2), (2, 3), (3, 4)])
        pytest.raises(nx.NetworkXNotImplemented, f, G)
        G = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
        pytest.raises(nx.NetworkXUnfeasible, f, G)

    def test_lexicographical_topological_sort(self):
        G = nx.DiGraph([(1, 2), (2, 3), (1, 4), (1, 5), (2, 6)])
        assert list(nx.lexicographical_topological_sort(G)) == [1, 2, 3, 4, 5, 6]
        assert list(nx.lexicographical_topological_sort(G, key=lambda x: x)) == [1, 2, 3, 4, 5, 6]
        assert list(nx.lexicographical_topological_sort(G, key=lambda x: -x)) == [1, 5, 4, 2, 6, 3]

    def test_lexicographical_topological_sort2(self):
        """
        Check the case of two or more nodes with same key value.
        Want to avoid exception raised due to comparing nodes directly.
        See Issue #3493
        """

        class Test_Node:

            def __init__(self, n):
                self.label = n
                self.priority = 1

            def __repr__(self):
                return f'Node({self.label})'

        def sorting_key(node):
            return node.priority
        test_nodes = [Test_Node(n) for n in range(4)]
        G = nx.DiGraph()
        edges = [(0, 1), (0, 2), (0, 3), (2, 3)]
        G.add_edges_from(((test_nodes[a], test_nodes[b]) for a, b in edges))
        sorting = list(nx.lexicographical_topological_sort(G, key=sorting_key))
        assert sorting == test_nodes