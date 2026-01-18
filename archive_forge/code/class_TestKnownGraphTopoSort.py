import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
class TestKnownGraphTopoSort(TestCaseWithKnownGraph):

    def assertTopoSortOrder(self, ancestry):
        """Check topo_sort and iter_topo_order is genuinely topological order.

        For every child in the graph, check if it comes after all of it's
        parents.
        """
        graph = self.make_known_graph(ancestry)
        sort_result = graph.topo_sort()
        self.assertEqual(len(ancestry), len(sort_result))
        node_idx = {node: idx for idx, node in enumerate(sort_result)}
        for node in sort_result:
            parents = ancestry[node]
            for parent in parents:
                if parent not in ancestry:
                    continue
                if node_idx[node] <= node_idx[parent]:
                    self.fail('parent %s must come before child %s:\n%s' % (parent, node, sort_result))

    def test_topo_sort_empty(self):
        """TopoSort empty list"""
        self.assertTopoSortOrder({})

    def test_topo_sort_easy(self):
        """TopoSort list with one node"""
        self.assertTopoSortOrder({0: []})

    def test_topo_sort_cycle(self):
        """TopoSort traps graph with cycles"""
        g = self.make_known_graph({0: [1], 1: [0]})
        self.assertRaises(errors.GraphCycleError, g.topo_sort)

    def test_topo_sort_cycle_2(self):
        """TopoSort traps graph with longer cycle"""
        g = self.make_known_graph({0: [1], 1: [2], 2: [0]})
        self.assertRaises(errors.GraphCycleError, g.topo_sort)

    def test_topo_sort_cycle_with_tail(self):
        """TopoSort traps graph with longer cycle"""
        g = self.make_known_graph({0: [1], 1: [2], 2: [3, 4], 3: [0], 4: []})
        self.assertRaises(errors.GraphCycleError, g.topo_sort)

    def test_topo_sort_1(self):
        """TopoSort simple nontrivial graph"""
        self.assertTopoSortOrder({0: [3], 1: [4], 2: [1, 4], 3: [], 4: [0, 3]})

    def test_topo_sort_partial(self):
        """Topological sort with partial ordering.

        Multiple correct orderings are possible, so test for
        correctness, not for exact match on the resulting list.
        """
        self.assertTopoSortOrder({0: [], 1: [0], 2: [0], 3: [0], 4: [1, 2, 3], 5: [1, 2], 6: [1, 2], 7: [2, 3], 8: [0, 1, 4, 5, 6]})

    def test_topo_sort_ghost_parent(self):
        """Sort nodes, but don't include some parents in the output"""
        self.assertTopoSortOrder({0: [1], 1: [2]})