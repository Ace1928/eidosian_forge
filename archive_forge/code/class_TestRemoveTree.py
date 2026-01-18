import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
class TestRemoveTree(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.tree = self.make_branch_and_tree('branch1')
        self.build_tree(['branch1/foo'])
        self.tree.add('foo')
        self.tree.commit('1')
        self.assertPathExists('branch1/foo')

    def test_remove_tree_original_branch(self):
        self.run_bzr('remove-tree', working_dir='branch1')
        self.assertPathDoesNotExist('branch1/foo')

    def test_remove_tree_original_branch_explicit(self):
        self.run_bzr('remove-tree branch1')
        self.assertPathDoesNotExist('branch1/foo')

    def test_remove_tree_multiple_branch_explicit(self):
        self.tree.controldir.sprout('branch2')
        self.run_bzr('remove-tree branch1 branch2')
        self.assertPathDoesNotExist('branch1/foo')
        self.assertPathDoesNotExist('branch2/foo')

    def test_remove_tree_sprouted_branch(self):
        self.tree.controldir.sprout('branch2')
        self.assertPathExists('branch2/foo')
        self.run_bzr('remove-tree', working_dir='branch2')
        self.assertPathDoesNotExist('branch2/foo')

    def test_remove_tree_sprouted_branch_explicit(self):
        self.tree.controldir.sprout('branch2')
        self.assertPathExists('branch2/foo')
        self.run_bzr('remove-tree branch2')
        self.assertPathDoesNotExist('branch2/foo')

    def test_remove_tree_checkout(self):
        self.tree.branch.create_checkout('branch2', lightweight=False)
        self.assertPathExists('branch2/foo')
        self.run_bzr('remove-tree', working_dir='branch2')
        self.assertPathDoesNotExist('branch2/foo')
        self.assertPathExists('branch1/foo')

    def test_remove_tree_checkout_explicit(self):
        self.tree.branch.create_checkout('branch2', lightweight=False)
        self.assertPathExists('branch2/foo')
        self.run_bzr('remove-tree branch2')
        self.assertPathDoesNotExist('branch2/foo')
        self.assertPathExists('branch1/foo')

    def test_remove_tree_lightweight_checkout(self):
        self.tree.branch.create_checkout('branch2', lightweight=True)
        self.assertPathExists('branch2/foo')
        output = self.run_bzr_error(['You cannot remove the working tree from a lightweight checkout'], 'remove-tree', retcode=3, working_dir='branch2')
        self.assertPathExists('branch2/foo')
        self.assertPathExists('branch1/foo')

    def test_remove_tree_lightweight_checkout_explicit(self):
        self.tree.branch.create_checkout('branch2', lightweight=True)
        self.assertPathExists('branch2/foo')
        output = self.run_bzr_error(['You cannot remove the working tree from a lightweight checkout'], 'remove-tree branch2', retcode=3)
        self.assertPathExists('branch2/foo')
        self.assertPathExists('branch1/foo')

    def test_remove_tree_empty_dir(self):
        os.mkdir('branch2')
        output = self.run_bzr_error(['Not a branch'], 'remove-tree', retcode=3, working_dir='branch2')

    def test_remove_tree_repeatedly(self):
        self.run_bzr('remove-tree branch1')
        self.assertPathDoesNotExist('branch1/foo')
        output = self.run_bzr_error(['No working tree to remove'], 'remove-tree branch1', retcode=3)

    def test_remove_tree_remote_path(self):
        pass

    def test_remove_tree_uncommitted_changes(self):
        self.build_tree(['branch1/bar'])
        self.tree.add('bar')
        output = self.run_bzr_error(['Working tree .* has uncommitted changes'], 'remove-tree branch1', retcode=3)

    def test_remove_tree_uncommitted_changes_force(self):
        self.build_tree(['branch1/bar'])
        self.tree.add('bar')
        self.run_bzr('remove-tree branch1 --force')
        self.assertPathDoesNotExist('branch1/foo')
        self.assertPathExists('branch1/bar')

    def test_remove_tree_pending_merges(self):
        self.run_bzr(['branch', 'branch1', 'branch2'])
        self.build_tree(['branch1/bar'])
        self.tree.add('bar')
        self.tree.commit('2')
        self.assertPathExists('branch1/bar')
        self.run_bzr(['merge', '../branch1'], working_dir='branch2')
        self.assertPathExists('branch2/bar')
        self.run_bzr(['revert', '.'], working_dir='branch2')
        self.assertPathDoesNotExist('branch2/bar')
        output = self.run_bzr_error(['Working tree .* has uncommitted changes'], 'remove-tree branch2', retcode=3)

    def test_remove_tree_pending_merges_force(self):
        self.run_bzr(['branch', 'branch1', 'branch2'])
        self.build_tree(['branch1/bar'])
        self.tree.add('bar')
        self.tree.commit('2')
        self.assertPathExists('branch1/bar')
        self.run_bzr(['merge', '../branch1'], working_dir='branch2')
        self.assertPathExists('branch2/bar')
        self.run_bzr(['revert', '.'], working_dir='branch2')
        self.assertPathDoesNotExist('branch2/bar')
        self.run_bzr('remove-tree branch2 --force')
        self.assertPathDoesNotExist('branch2/foo')
        self.assertPathDoesNotExist('branch2/bar')

    def test_remove_tree_shelved_changes(self):
        tree = self.make_branch_and_tree('.')
        creator = shelf.ShelfCreator(tree, tree.basis_tree(), [])
        self.addCleanup(creator.finalize)
        shelf_id = tree.get_shelf_manager().shelve_changes(creator, 'Foo')
        output = self.run_bzr_error(['Working tree .* has shelved changes'], 'remove-tree', retcode=3)

    def test_remove_tree_shelved_changes_force(self):
        tree = self.make_branch_and_tree('.')
        creator = shelf.ShelfCreator(tree, tree.basis_tree(), [])
        self.addCleanup(creator.finalize)
        shelf_id = tree.get_shelf_manager().shelve_changes(creator, 'Foo')
        self.run_bzr('remove-tree --force')
        self.run_bzr('checkout')
        self.assertIs(None, tree.get_shelf_manager().last_shelf())