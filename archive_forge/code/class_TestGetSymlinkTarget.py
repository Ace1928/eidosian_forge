import os
from breezy import osutils, tests
from breezy.tests import features, per_tree
class TestGetSymlinkTarget(per_tree.TestCaseWithTree):

    def get_tree_with_symlinks(self):
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        tree = self.make_branch_and_tree('tree')
        os.symlink('foo', 'tree/link')
        os.symlink('../bar', 'tree/rel_link')
        os.symlink('/baz/bing', 'tree/abs_link')
        tree.add(['link', 'rel_link', 'abs_link'])
        return self._convert_tree(tree)

    def test_get_symlink_target(self):
        tree = self.get_tree_with_symlinks()
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual('foo', tree.get_symlink_target('link'))
        self.assertEqual('../bar', tree.get_symlink_target('rel_link'))
        self.assertEqual('/baz/bing', tree.get_symlink_target('abs_link'))
        self.assertEqual('foo', tree.get_symlink_target('link'))

    def test_get_unicode_symlink_target(self):
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        self.requireFeature(features.UnicodeFilenameFeature)
        tree = self.make_branch_and_tree('tree')
        target = 'targ€t'
        os.symlink(target, os.fsencode('tree/β_link'))
        tree.add(['β_link'])
        tree.lock_read()
        self.addCleanup(tree.unlock)
        actual = tree.get_symlink_target('β_link')
        self.assertEqual(target, actual)