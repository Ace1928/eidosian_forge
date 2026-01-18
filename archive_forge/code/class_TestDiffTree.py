import contextlib
import os
import re
import subprocess
import sys
import tempfile
from io import BytesIO
from .. import diff, errors, osutils
from .. import revision as _mod_revision
from .. import revisionspec, revisiontree, tests
from ..tests import EncodingAdapter, features
from ..tests.scenarios import load_tests_apply_scenarios
class TestDiffTree(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.old_tree = self.make_branch_and_tree('old-tree')
        self.old_tree.lock_write()
        self.addCleanup(self.old_tree.unlock)
        self.new_tree = self.make_branch_and_tree('new-tree')
        self.new_tree.lock_write()
        self.addCleanup(self.new_tree.unlock)
        self.differ = diff.DiffTree(self.old_tree, self.new_tree, BytesIO())

    def test_diff_text(self):
        self.build_tree_contents([('old-tree/olddir/',), ('old-tree/olddir/oldfile', b'old\n')])
        self.old_tree.add('olddir')
        self.old_tree.add('olddir/oldfile', ids=b'file-id')
        self.build_tree_contents([('new-tree/newdir/',), ('new-tree/newdir/newfile', b'new\n')])
        self.new_tree.add('newdir')
        self.new_tree.add('newdir/newfile', ids=b'file-id')
        differ = diff.DiffText(self.old_tree, self.new_tree, BytesIO())
        differ.diff_text('olddir/oldfile', None, 'old label', 'new label')
        self.assertEqual(b'--- old label\n+++ new label\n@@ -1,1 +0,0 @@\n-old\n\n', differ.to_file.getvalue())
        differ.to_file.seek(0)
        differ.diff_text(None, 'newdir/newfile', 'old label', 'new label')
        self.assertEqual(b'--- old label\n+++ new label\n@@ -0,0 +1,1 @@\n+new\n\n', differ.to_file.getvalue())
        differ.to_file.seek(0)
        differ.diff_text('olddir/oldfile', 'newdir/newfile', 'old label', 'new label')
        self.assertEqual(b'--- old label\n+++ new label\n@@ -1,1 +1,1 @@\n-old\n+new\n\n', differ.to_file.getvalue())

    def test_diff_deletion(self):
        self.build_tree_contents([('old-tree/file', b'contents'), ('new-tree/file', b'contents')])
        self.old_tree.add('file', ids=b'file-id')
        self.new_tree.add('file', ids=b'file-id')
        os.unlink('new-tree/file')
        self.differ.show_diff(None)
        self.assertContainsRe(self.differ.to_file.getvalue(), b'-contents')

    def test_diff_creation(self):
        self.build_tree_contents([('old-tree/file', b'contents'), ('new-tree/file', b'contents')])
        self.old_tree.add('file', ids=b'file-id')
        self.new_tree.add('file', ids=b'file-id')
        os.unlink('old-tree/file')
        self.differ.show_diff(None)
        self.assertContainsRe(self.differ.to_file.getvalue(), b'\\+contents')

    def test_diff_symlink(self):
        differ = diff.DiffSymlink(self.old_tree, self.new_tree, BytesIO())
        differ.diff_symlink('old target', None)
        self.assertEqual(b"=== target was 'old target'\n", differ.to_file.getvalue())
        differ = diff.DiffSymlink(self.old_tree, self.new_tree, BytesIO())
        differ.diff_symlink(None, 'new target')
        self.assertEqual(b"=== target is 'new target'\n", differ.to_file.getvalue())
        differ = diff.DiffSymlink(self.old_tree, self.new_tree, BytesIO())
        differ.diff_symlink('old target', 'new target')
        self.assertEqual(b"=== target changed 'old target' => 'new target'\n", differ.to_file.getvalue())

    def test_diff(self):
        self.build_tree_contents([('old-tree/olddir/',), ('old-tree/olddir/oldfile', b'old\n')])
        self.old_tree.add('olddir')
        self.old_tree.add('olddir/oldfile', ids=b'file-id')
        self.build_tree_contents([('new-tree/newdir/',), ('new-tree/newdir/newfile', b'new\n')])
        self.new_tree.add('newdir')
        self.new_tree.add('newdir/newfile', ids=b'file-id')
        self.differ.diff('olddir/oldfile', 'newdir/newfile')
        self.assertContainsRe(self.differ.to_file.getvalue(), b'--- olddir/oldfile.*\\n\\+\\+\\+ newdir/newfile.*\\n\\@\\@ -1,1 \\+1,1 \\@\\@\\n-old\\n\\+new\\n\\n')

    def test_diff_kind_change(self):
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        self.build_tree_contents([('old-tree/olddir/',), ('old-tree/olddir/oldfile', b'old\n')])
        self.old_tree.add('olddir')
        self.old_tree.add('olddir/oldfile', ids=b'file-id')
        self.build_tree(['new-tree/newdir/'])
        os.symlink('new', 'new-tree/newdir/newfile')
        self.new_tree.add('newdir')
        self.new_tree.add('newdir/newfile', ids=b'file-id')
        self.differ.diff('olddir/oldfile', 'newdir/newfile')
        self.assertContainsRe(self.differ.to_file.getvalue(), b'--- olddir/oldfile.*\\n\\+\\+\\+ newdir/newfile.*\\n\\@\\@ -1,1 \\+0,0 \\@\\@\\n-old\\n\\n')
        self.assertContainsRe(self.differ.to_file.getvalue(), b"=== target is 'new'\n")

    def test_diff_directory(self):
        self.build_tree(['new-tree/new-dir/'])
        self.new_tree.add('new-dir', ids=b'new-dir-id')
        self.differ.diff(None, 'new-dir')
        self.assertEqual(self.differ.to_file.getvalue(), b'')

    def create_old_new(self):
        self.build_tree_contents([('old-tree/olddir/',), ('old-tree/olddir/oldfile', b'old\n')])
        self.old_tree.add('olddir')
        self.old_tree.add('olddir/oldfile', ids=b'file-id')
        self.build_tree_contents([('new-tree/newdir/',), ('new-tree/newdir/newfile', b'new\n')])
        self.new_tree.add('newdir')
        self.new_tree.add('newdir/newfile', ids=b'file-id')

    def test_register_diff(self):
        self.create_old_new()
        old_diff_factories = diff.DiffTree.diff_factories
        diff.DiffTree.diff_factories = old_diff_factories[:]
        diff.DiffTree.diff_factories.insert(0, DiffWasIs.from_diff_tree)
        try:
            differ = diff.DiffTree(self.old_tree, self.new_tree, BytesIO())
        finally:
            diff.DiffTree.diff_factories = old_diff_factories
        differ.diff('olddir/oldfile', 'newdir/newfile')
        self.assertNotContainsRe(differ.to_file.getvalue(), b'--- olddir/oldfile.*\\n\\+\\+\\+ newdir/newfile.*\\n\\@\\@ -1,1 \\+1,1 \\@\\@\\n-old\\n\\+new\\n\\n')
        self.assertContainsRe(differ.to_file.getvalue(), b'was: old\nis: new\n')

    def test_extra_factories(self):
        self.create_old_new()
        differ = diff.DiffTree(self.old_tree, self.new_tree, BytesIO(), extra_factories=[DiffWasIs.from_diff_tree])
        differ.diff('olddir/oldfile', 'newdir/newfile')
        self.assertNotContainsRe(differ.to_file.getvalue(), b'--- olddir/oldfile.*\\n\\+\\+\\+ newdir/newfile.*\\n\\@\\@ -1,1 \\+1,1 \\@\\@\\n-old\\n\\+new\\n\\n')
        self.assertContainsRe(differ.to_file.getvalue(), b'was: old\nis: new\n')

    def test_alphabetical_order(self):
        self.build_tree(['new-tree/a-file'])
        self.new_tree.add('a-file')
        self.build_tree(['old-tree/b-file'])
        self.old_tree.add('b-file')
        self.differ.show_diff(None)
        self.assertContainsRe(self.differ.to_file.getvalue(), b'.*a-file(.|\n)*b-file')