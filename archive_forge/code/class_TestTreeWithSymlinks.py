from breezy import osutils, tests
from breezy.git.branch import GitBranch
from breezy.mutabletree import MutableTree
from breezy.tests import TestSkipped, features, per_tree
from breezy.transform import PreviewTree
class TestTreeWithSymlinks(per_tree.TestCaseWithTree):

    def setUp(self):
        super().setUp()
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        self.tree = self.get_tree_with_subdirs_and_all_content_types()
        self.tree.lock_read()
        self.addCleanup(self.tree.unlock)

    def test_symlink_target(self):
        if isinstance(self.tree, (MutableTree, PreviewTree)):
            raise TestSkipped('symlinks not accurately represented in working trees and preview trees')
        entry = get_entry(self.tree, 'symlink')
        self.assertEqual(entry.symlink_target, 'link-target')

    def test_symlink_target_tree(self):
        self.assertEqual('link-target', self.tree.get_symlink_target('symlink'))

    def test_kind_symlink(self):
        self.assertEqual('symlink', self.tree.kind('symlink'))
        self.assertIs(None, self.tree.get_file_size('symlink'))

    def test_symlink(self):
        entry = get_entry(self.tree, 'symlink')
        self.assertEqual(entry.kind, 'symlink')
        self.assertEqual(None, entry.text_size)