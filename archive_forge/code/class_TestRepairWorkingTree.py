from breezy import workingtree
from breezy.tests import TestCaseWithTransport
class TestRepairWorkingTree(TestCaseWithTransport):

    def break_dirstate(self, tree, completely=False):
        """Write garbage into the dirstate file."""
        self.assertIsNot(None, getattr(tree, 'current_dirstate', None))
        with tree.lock_read():
            dirstate = tree.current_dirstate()
            dirstate_path = dirstate._filename
            self.assertPathExists(dirstate_path)
        if completely:
            f = open(dirstate_path, 'wb')
        else:
            f = open(dirstate_path, 'ab')
        try:
            f.write(b'garbage-at-end-of-file\n')
        finally:
            f.close()

    def make_initial_tree(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/foo', 'tree/dir/', 'tree/dir/bar'])
        tree.add(['foo', 'dir', 'dir/bar'])
        tree.commit('first')
        return tree

    def test_repair_refuses_uncorrupted(self):
        tree = self.make_initial_tree()
        self.run_bzr_error(['The tree does not appear to be corrupt', '"brz revert"', '--force'], 'repair-workingtree -d tree')

    def test_repair_forced(self):
        tree = self.make_initial_tree()
        tree.rename_one('dir', 'alt_dir')
        self.assertTrue(tree.is_versioned('alt_dir'))
        self.run_bzr('repair-workingtree -d tree --force')
        self.assertFalse(tree.is_versioned('alt_dir'))
        self.assertPathExists('tree/alt_dir')

    def test_repair_corrupted_dirstate(self):
        tree = self.make_initial_tree()
        self.break_dirstate(tree)
        self.run_bzr('repair-workingtree -d tree')
        tree = workingtree.WorkingTree.open('tree')
        tree.check_state()

    def test_repair_naive_destroyed_fails(self):
        tree = self.make_initial_tree()
        self.break_dirstate(tree, completely=True)
        self.run_bzr_error(['the header appears corrupt, try passing'], 'repair-workingtree -d tree')

    def test_repair_destroyed_with_revs_passes(self):
        tree = self.make_initial_tree()
        self.break_dirstate(tree, completely=True)
        self.run_bzr('repair-workingtree -d tree -r -1')
        tree = workingtree.WorkingTree.open('tree')
        tree.check_state()