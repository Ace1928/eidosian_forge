from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
class TestGraphFindDistanceToNull(TestGraphBase):
    """Test an api that should be able to compute a revno"""

    def assertFindDistance(self, revno, graph, target_id, known_ids):
        """Assert the output of Graph.find_distance_to_null()"""
        actual = graph.find_distance_to_null(target_id, known_ids)
        self.assertEqual(revno, actual)

    def test_nothing_known(self):
        graph = self.make_graph(ancestry_1)
        self.assertFindDistance(0, graph, NULL_REVISION, [])
        self.assertFindDistance(1, graph, b'rev1', [])
        self.assertFindDistance(2, graph, b'rev2a', [])
        self.assertFindDistance(2, graph, b'rev2b', [])
        self.assertFindDistance(3, graph, b'rev3', [])
        self.assertFindDistance(4, graph, b'rev4', [])

    def test_rev_is_ghost(self):
        graph = self.make_graph(ancestry_1)
        e = self.assertRaises(errors.GhostRevisionsHaveNoRevno, graph.find_distance_to_null, b'rev_missing', [])
        self.assertEqual(b'rev_missing', e.revision_id)
        self.assertEqual(b'rev_missing', e.ghost_revision_id)

    def test_ancestor_is_ghost(self):
        graph = self.make_graph({b'rev': [b'parent']})
        e = self.assertRaises(errors.GhostRevisionsHaveNoRevno, graph.find_distance_to_null, b'rev', [])
        self.assertEqual(b'rev', e.revision_id)
        self.assertEqual(b'parent', e.ghost_revision_id)

    def test_known_in_ancestry(self):
        graph = self.make_graph(ancestry_1)
        self.assertFindDistance(2, graph, b'rev2a', [(b'rev1', 1)])
        self.assertFindDistance(3, graph, b'rev3', [(b'rev2a', 2)])

    def test_known_in_ancestry_limits(self):
        graph = self.make_breaking_graph(ancestry_1, [b'rev1'])
        self.assertFindDistance(4, graph, b'rev4', [(b'rev3', 3)])

    def test_target_is_ancestor(self):
        graph = self.make_graph(ancestry_1)
        self.assertFindDistance(2, graph, b'rev2a', [(b'rev3', 3)])

    def test_target_is_ancestor_limits(self):
        """We shouldn't search all history if we run into ourselves"""
        graph = self.make_breaking_graph(ancestry_1, [b'rev1'])
        self.assertFindDistance(3, graph, b'rev3', [(b'rev4', 4)])

    def test_target_parallel_to_known_limits(self):
        graph = self.make_breaking_graph(with_tail, [b'a'])
        self.assertFindDistance(6, graph, b'f', [(b'g', 6)])
        self.assertFindDistance(7, graph, b'h', [(b'g', 6)])
        self.assertFindDistance(8, graph, b'i', [(b'g', 6)])
        self.assertFindDistance(6, graph, b'g', [(b'i', 8)])