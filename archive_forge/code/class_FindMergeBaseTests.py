from dulwich.tests import TestCase
from ..graph import WorkList, _find_lcas, can_fast_forward
from ..repo import MemoryRepo
from .utils import make_commit
class FindMergeBaseTests(TestCase):

    @staticmethod
    def run_test(dag, inputs):

        def lookup_parents(commit_id):
            return dag[commit_id]

        def lookup_stamp(commit_id):
            return 100
        c1 = inputs[0]
        c2s = inputs[1:]
        return set(_find_lcas(lookup_parents, c1, c2s, lookup_stamp))

    def test_multiple_lca(self):
        graph = {'5': ['1', '2'], '4': ['3', '1'], '3': ['2'], '2': ['0'], '1': [], '0': []}
        self.assertEqual(self.run_test(graph, ['4', '5']), {'1', '2'})

    def test_no_common_ancestor(self):
        graph = {'4': ['2'], '3': ['1'], '2': [], '1': ['0'], '0': []}
        self.assertEqual(self.run_test(graph, ['4', '3']), set())

    def test_ancestor(self):
        graph = {'G': ['D', 'F'], 'F': ['E'], 'D': ['C'], 'C': ['B'], 'E': ['B'], 'B': ['A'], 'A': []}
        self.assertEqual(self.run_test(graph, ['D', 'C']), {'C'})

    def test_direct_parent(self):
        graph = {'G': ['D', 'F'], 'F': ['E'], 'D': ['C'], 'C': ['B'], 'E': ['B'], 'B': ['A'], 'A': []}
        self.assertEqual(self.run_test(graph, ['G', 'D']), {'D'})

    def test_another_crossover(self):
        graph = {'G': ['D', 'F'], 'F': ['E', 'C'], 'D': ['C', 'E'], 'C': ['B'], 'E': ['B'], 'B': ['A'], 'A': []}
        self.assertEqual(self.run_test(graph, ['D', 'F']), {'E', 'C'})

    def test_three_way_merge_lca(self):
        graph = {'C': ['C1'], 'C1': ['C2'], 'C2': ['C3'], 'C3': ['C4'], 'C4': ['2'], 'B': ['B1'], 'B1': ['B2'], 'B2': ['B3'], 'B3': ['1'], 'A': ['A1'], 'A1': ['A2'], 'A2': ['A3'], 'A3': ['1'], '1': ['2'], '2': []}
        self.assertEqual(self.run_test(graph, ['A', 'B', 'C']), {'1'})

    def test_octopus(self):
        graph = {'C': ['C1'], 'C1': ['C2'], 'C2': ['C3'], 'C3': ['C4'], 'C4': ['2'], 'B': ['B1'], 'B1': ['B2'], 'B2': ['B3'], 'B3': ['1'], 'A': ['A1'], 'A1': ['A2'], 'A2': ['A3'], 'A3': ['1'], '1': ['2'], '2': []}

        def lookup_parents(cid):
            return graph[cid]

        def lookup_stamp(commit_id):
            return 100
        lcas = ['A']
        others = ['B', 'C']
        for cmt in others:
            next_lcas = []
            for ca in lcas:
                res = _find_lcas(lookup_parents, cmt, [ca], lookup_stamp)
                next_lcas.extend(res)
            lcas = next_lcas[:]
        self.assertEqual(set(lcas), {'2'})