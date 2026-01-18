import os
from breezy import osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
class TestExecutable(TestCaseWithWorkingTree):

    def setUp(self):
        super().setUp()
        self.a_id = b'a-20051208024829-849e76f7968d7a86'
        self.b_id = b'b-20051208024829-849e76f7968d7a86'
        wt = self.make_branch_and_tree('b1')
        b = wt.branch
        tt = wt.transform()
        tt.new_file('a', tt.root, [b'a test\n'], self.a_id, True)
        tt.new_file('b', tt.root, [b'b test\n'], self.b_id, False)
        tt.apply()
        self.wt = wt

    def check_exist(self, tree):
        """Just check that both files have the right executable bits set"""
        tree.lock_read()
        self.assertTrue(tree.is_executable('a'), "'a' lost the execute bit")
        self.assertFalse(tree.is_executable('b'), "'b' gained an execute bit")
        tree.unlock()

    def check_empty(self, tree, ignore_inv=False):
        """Check that the files are truly missing
        :param ignore_inv: If you just delete files from a working tree
                the inventory still shows them, so don't assert that
                the inventory is empty, just that the tree doesn't have them
        """
        with tree.lock_read():
            if not ignore_inv and getattr(tree, 'root_inventory', None):
                self.assertEqual([('', tree.root_inventory.root)], list(tree.root_inventory.iter_entries()))
            self.assertFalse(tree.has_filename('a'))
            self.assertFalse(tree.has_filename('b'))

    def commit_and_branch(self):
        """Commit the current tree, and create a second tree"""
        r1 = self.wt.commit('adding a,b')
        dir2 = self.wt.branch.controldir.sprout('b2', revision_id=r1)
        wt2 = dir2.open_workingtree()
        self.assertEqual([r1], wt2.get_parent_ids())
        self.assertEqual(r1, wt2.branch.last_revision())
        return (wt2, r1)

    def test_01_is_executable(self):
        """Make sure that the tree was created and has the executable bit set"""
        self.check_exist(self.wt)

    def test_02_stays_executable(self):
        """reopen the tree and ensure it stuck."""
        self.wt = self.wt.controldir.open_workingtree()
        self.check_exist(self.wt)

    def test_03_after_commit(self):
        """Commit the change, and check the history"""
        r1 = self.wt.commit('adding a,b')
        rev_tree = self.wt.branch.repository.revision_tree(r1)
        self.check_exist(rev_tree)

    def test_04_after_removed(self):
        """Make sure reverting removed files brings them back correctly"""
        r1 = self.wt.commit('adding a,b')
        os.remove('b1/a')
        os.remove('b1/b')
        self.check_empty(self.wt, ignore_inv=True)
        rev_tree = self.wt.branch.repository.revision_tree(r1)
        self.wt.revert(['a', 'b'], rev_tree, backups=False)
        self.check_exist(self.wt)

    def test_05_removed_and_committed(self):
        """Check that reverting to an earlier commit restores them"""
        r1 = self.wt.commit('adding a,b')
        os.remove('b1/a')
        os.remove('b1/b')
        r2 = self.wt.commit('removed')
        self.check_empty(self.wt)
        rev_tree = self.wt.branch.repository.revision_tree(r1)
        self.wt.revert(old_tree=rev_tree, backups=False)
        self.check_exist(self.wt)

    def test_06_branch(self):
        """branch b1=>b2 should preserve the executable bits"""
        wt2, r1 = self.commit_and_branch()
        self.check_exist(wt2)

    def test_07_pull(self):
        """Test that pull will handle bits correctly"""
        wt2, r1 = self.commit_and_branch()
        os.remove('b1/a')
        os.remove('b1/b')
        r2 = self.wt.commit('removed')
        wt2.pull(self.wt.branch)
        self.assertEqual([r2], wt2.get_parent_ids())
        self.assertEqual(r2, wt2.branch.last_revision())
        self.check_empty(wt2)
        rev_tree = self.wt.branch.repository.revision_tree(r1)
        self.wt.revert(old_tree=rev_tree, backups=False)
        r3 = self.wt.commit('resurrected')
        self.check_exist(self.wt)
        wt2.pull(self.wt.branch)
        self.assertEqual([r3], wt2.get_parent_ids())
        self.assertEqual(r3, wt2.branch.last_revision())
        self.check_exist(wt2)

    def test_08_no_op_revert(self):
        """Just do a simple revert without anything changed

        The bits shouldn't swap.
        """
        r1 = self.wt.commit('adding a,b')
        rev_tree = self.wt.branch.repository.revision_tree(r1)
        self.wt.revert(old_tree=rev_tree, backups=False)
        self.check_exist(self.wt)

    def test_commit_with_exec_from_basis(self):
        self.wt._is_executable_from_path_and_stat = self.wt._is_executable_from_path_and_stat_from_basis
        rev_id1 = self.wt.commit('one')
        rev_tree1 = self.wt.branch.repository.revision_tree(rev_id1)
        a_executable = rev_tree1.is_executable('a')
        b_executable = rev_tree1.is_executable('b')
        self.assertIsNot(None, a_executable)
        self.assertTrue(a_executable)
        self.assertIsNot(None, b_executable)
        self.assertFalse(b_executable)

    def test_use_exec_from_basis(self):
        self.wt._supports_executable = lambda: False
        self.addCleanup(self.wt.lock_read().unlock)
        self.assertTrue(self.wt.is_executable('a'))
        self.assertFalse(self.wt.is_executable('b'))