import pprint
from breezy.errors import GraphCycleError
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase
from breezy.tsort import MergeSorter, TopoSorter, merge_sort, topo_sort
class TopoSortTests(TestCase):

    def assertSortAndIterate(self, graph, result_list):
        """Check that sorting and iter_topo_order on graph works."""
        self.assertEqual(result_list, topo_sort(graph))
        self.assertEqual(result_list, list(TopoSorter(graph).iter_topo_order()))

    def assertSortAndIterateRaise(self, exception_type, graph):
        """Try iterating and topo_sorting graph and expect an exception."""
        self.assertRaises(exception_type, topo_sort, graph)
        self.assertRaises(exception_type, list, TopoSorter(graph).iter_topo_order())

    def assertSortAndIterateOrder(self, graph):
        """Check topo_sort and iter_topo_order is genuinely topological order.

        For every child in the graph, check if it comes after all of it's
        parents.
        """
        sort_result = topo_sort(graph)
        iter_result = list(TopoSorter(graph).iter_topo_order())
        for node, parents in graph:
            for parent in parents:
                if sort_result.index(node) < sort_result.index(parent):
                    self.fail('parent %s must come before child %s:\n%s' % (parent, node, sort_result))
                if iter_result.index(node) < iter_result.index(parent):
                    self.fail('parent %s must come before child %s:\n%s' % (parent, node, iter_result))

    def test_tsort_empty(self):
        """TopoSort empty list"""
        self.assertSortAndIterate([], [])

    def test_tsort_easy(self):
        """TopoSort list with one node"""
        self.assertSortAndIterate({0: []}.items(), [0])

    def test_tsort_cycle(self):
        """TopoSort traps graph with cycles"""
        self.assertSortAndIterateRaise(GraphCycleError, {0: [1], 1: [0]}.items())

    def test_tsort_cycle_2(self):
        """TopoSort traps graph with longer cycle"""
        self.assertSortAndIterateRaise(GraphCycleError, {0: [1], 1: [2], 2: [0]}.items())

    def test_topo_sort_cycle_with_tail(self):
        """TopoSort traps graph with longer cycle"""
        self.assertSortAndIterateRaise(GraphCycleError, {0: [1], 1: [2], 2: [3, 4], 3: [0], 4: []}.items())

    def test_tsort_1(self):
        """TopoSort simple nontrivial graph"""
        self.assertSortAndIterate({0: [3], 1: [4], 2: [1, 4], 3: [], 4: [0, 3]}.items(), [3, 0, 4, 1, 2])

    def test_tsort_partial(self):
        """Topological sort with partial ordering.

        Multiple correct orderings are possible, so test for
        correctness, not for exact match on the resulting list.
        """
        self.assertSortAndIterateOrder([(0, []), (1, [0]), (2, [0]), (3, [0]), (4, [1, 2, 3]), (5, [1, 2]), (6, [1, 2]), (7, [2, 3]), (8, [0, 1, 4, 5, 6])])

    def test_tsort_unincluded_parent(self):
        """Sort nodes, but don't include some parents in the output"""
        self.assertSortAndIterate([(0, [1]), (1, [2])], [1, 0])