import os
from ..workspace import Workspace, WorkspaceDirty, check_clean_tree
from . import TestCaseWithTransport, features, multiply_scenarios
from .scenarios import load_tests_apply_scenarios
class CheckCleanTreeTests(TestCaseWithTransport):

    def make_test_tree(self, format=None):
        tree = self.make_branch_and_tree('.', format=format)
        self.build_tree_contents([('debian/',), ('debian/control', 'Source: blah\nVcs-Git: https://example.com/blah\nTestsuite: autopkgtest\n\nBinary: blah\nArch: all\n\n'), ('debian/changelog', 'Some contents')])
        tree.add(['debian', 'debian/changelog', 'debian/control'])
        tree.commit('Initial thingy.')
        return tree

    def test_pending_changes(self):
        tree = self.make_test_tree()
        self.build_tree_contents([('debian/changelog', 'blah')])
        with tree.lock_write():
            self.assertRaises(WorkspaceDirty, check_clean_tree, tree)

    def test_pending_changes_bzr_empty_dir(self):
        tree = self.make_test_tree(format='bzr')
        self.build_tree_contents([('debian/upstream/',)])
        with tree.lock_write():
            self.assertRaises(WorkspaceDirty, check_clean_tree, tree)

    def test_pending_changes_git_empty_dir(self):
        tree = self.make_test_tree(format='git')
        self.build_tree_contents([('debian/upstream/',)])
        with tree.lock_write():
            check_clean_tree(tree)

    def test_pending_changes_git_dir_with_ignored(self):
        tree = self.make_test_tree(format='git')
        self.build_tree_contents([('debian/upstream/',), ('debian/upstream/blah', ''), ('.gitignore', 'blah\n')])
        tree.add('.gitignore')
        tree.commit('add gitignore')
        with tree.lock_write():
            check_clean_tree(tree)

    def test_extra(self):
        tree = self.make_test_tree()
        self.build_tree_contents([('debian/foo', 'blah')])
        with tree.lock_write():
            self.assertRaises(WorkspaceDirty, check_clean_tree, tree)

    def test_subpath(self):
        tree = self.make_test_tree()
        self.build_tree_contents([('debian/foo', 'blah'), ('foo/',)])
        tree.add('foo')
        tree.commit('add foo')
        with tree.lock_write():
            check_clean_tree(tree, tree.basis_tree(), subpath='foo')
            self.assertRaises(WorkspaceDirty, check_clean_tree, tree, tree.basis_tree(), subpath='')

    def test_subpath_changed(self):
        tree = self.make_test_tree()
        self.build_tree_contents([('foo/',)])
        tree.add('foo')
        tree.commit('add foo')
        self.build_tree_contents([('debian/control', 'blah')])
        with tree.lock_write():
            check_clean_tree(tree, tree.basis_tree(), subpath='foo')
            self.assertRaises(WorkspaceDirty, check_clean_tree, tree, tree.basis_tree(), subpath='')