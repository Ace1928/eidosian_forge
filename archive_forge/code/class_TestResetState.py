from breezy import errors, tests
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
class TestResetState(TestCaseWithState):

    def make_initial_tree(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/foo', 'tree/dir/', 'tree/dir/bar'])
        tree.add(['foo', 'dir', 'dir/bar'])
        tree.commit('initial')
        return tree

    def test_reset_state_forgets_changes(self):
        tree = self.make_initial_tree()
        tree.rename_one('foo', 'baz')
        self.assertFalse(tree.is_versioned('foo'))
        if tree.supports_rename_tracking() and tree.supports_file_ids:
            foo_id = tree.basis_tree().path2id('foo')
            self.assertEqual(foo_id, tree.path2id('baz'))
        else:
            self.assertTrue(tree.is_versioned('baz'))
        tree.reset_state()
        if tree.supports_file_ids:
            self.assertEqual(foo_id, tree.path2id('foo'))
            self.assertEqual(None, tree.path2id('baz'))
        self.assertPathDoesNotExist('tree/foo')
        self.assertPathExists('tree/baz')

    def test_reset_state_handles_corrupted_dirstate(self):
        tree = self.make_initial_tree()
        rev_id = tree.last_revision()
        self.break_dirstate(tree)
        tree.reset_state()
        tree.check_state()
        self.assertEqual(rev_id, tree.last_revision())

    def test_reset_state_handles_destroyed_dirstate(self):
        tree = self.make_initial_tree()
        rev_id = tree.last_revision()
        self.break_dirstate(tree, completely=True)
        tree.reset_state(revision_ids=[rev_id])
        tree.check_state()
        self.assertEqual(rev_id, tree.last_revision())