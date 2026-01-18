from typing import List, Tuple
from breezy import errors, revision
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tree import (FileTimestampUnavailable, InterTree,
class TestTree(TestCaseWithTransport):

    def test_compare_calls_InterTree_compare(self):
        """This test tests the way Tree.compare() uses InterTree."""
        old_optimisers = InterTree._optimisers
        try:
            InterTree._optimisers = []
            RecordingOptimiser.calls = []
            InterTree.register_optimiser(RecordingOptimiser)
            tree = self.make_branch_and_tree('1')
            null_tree = tree.basis_tree()
            tree2 = self.make_branch_and_tree('2')
            tree.changes_from(tree2)
            tree.changes_from(tree2, 'unchanged', 'specific', 'extra', 'require', True)
            tree.changes_from(tree2, specific_files='specific', want_unchanged='unchanged', extra_trees='extra', require_versioned='require', include_root=True, want_unversioned=True)
        finally:
            InterTree._optimisers = old_optimisers
        self.assertEqual([('find_source_path', null_tree, tree, '', 'none'), ('find_source_path', null_tree, tree2, '', 'none'), ('compare', tree2, tree, False, None, None, False, False, False), ('compare', tree2, tree, 'unchanged', 'specific', 'extra', 'require', True, False), ('compare', tree2, tree, 'unchanged', 'specific', 'extra', 'require', True, True)], RecordingOptimiser.calls)

    def test_changes_from_with_root(self):
        """Ensure the include_root option does what's expected."""
        wt = self.make_branch_and_tree('.')
        delta = wt.changes_from(wt.basis_tree())
        self.assertEqual(len(delta.added), 0)
        delta = wt.changes_from(wt.basis_tree(), include_root=True)
        self.assertEqual(len(delta.added), 1)
        self.assertEqual(delta.added[0].path[1], '')

    def test_changes_from_with_require_versioned(self):
        """Ensure the require_versioned option does what's expected."""
        wt = self.make_branch_and_tree('.')
        self.build_tree(['known_file', 'unknown_file'])
        wt.add('known_file')
        self.assertRaises(errors.PathsNotVersionedError, wt.changes_from, wt.basis_tree(), wt, specific_files=['known_file', 'unknown_file'], require_versioned=True)
        delta = wt.changes_from(wt.basis_tree(), specific_files=['known_file', 'unknown_file'], require_versioned=False)
        self.assertEqual(len(delta.added), 1)