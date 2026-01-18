import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
class AppliedPatchesTests(TestCaseWithTransport):

    def test_apply_simple(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('a', 'a\n')])
        tree.add('a')
        tree.commit('Add a')
        patch = parse_patch(b'--- a/a\n+++ a/a\n@@ -1 +1 @@\n-a\n+b\n'.splitlines(True))
        with AppliedPatches(tree, [patch]) as newtree:
            self.assertEqual(b'b\n', newtree.get_file_text('a'))

    def test_apply_delete(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('a', 'a\n')])
        tree.add('a')
        tree.commit('Add a')
        patch = parse_patch(b'--- a/a\n+++ /dev/null\n@@ -1 +0,0 @@\n-a\n'.splitlines(True))
        with AppliedPatches(tree, [patch]) as newtree:
            self.assertFalse(newtree.has_filename('a'))

    def test_apply_add(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('a', 'a\n')])
        tree.add('a')
        tree.commit('Add a')
        patch = parse_patch(b'--- /dev/null\n+++ a/b\n@@ -0,0 +1 @@\n+b\n'.splitlines(True))
        with AppliedPatches(tree, [patch]) as newtree:
            self.assertEqual(b'b\n', newtree.get_file_text('b'))