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
class TestDiffDates(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.wt = self.make_branch_and_tree('.')
        self.b = self.wt.branch
        self.build_tree_contents([('file1', b'file1 contents at rev 1\n'), ('file2', b'file2 contents at rev 1\n')])
        self.wt.add(['file1', 'file2'])
        self.wt.commit(message='Revision 1', timestamp=1143849600, timezone=0, rev_id=b'rev-1')
        self.build_tree_contents([('file1', b'file1 contents at rev 2\n')])
        self.wt.commit(message='Revision 2', timestamp=1143936000, timezone=28800, rev_id=b'rev-2')
        self.build_tree_contents([('file2', b'file2 contents at rev 3\n')])
        self.wt.commit(message='Revision 3', timestamp=1144022400, timezone=-3600, rev_id=b'rev-3')
        self.wt.remove(['file2'])
        self.wt.commit(message='Revision 4', timestamp=1144108800, timezone=0, rev_id=b'rev-4')
        self.build_tree_contents([('file1', b'file1 contents in working tree\n')])
        os.utime('file1', (1144195200, 1144195200))

    def test_diff_rev_tree_working_tree(self):
        output = get_diff_as_string(self.wt.basis_tree(), self.wt)
        self.assertEqualDiff(output, b"=== modified file 'file1'\n--- old/file1\t2006-04-02 00:00:00 +0000\n+++ new/file1\t2006-04-05 00:00:00 +0000\n@@ -1,1 +1,1 @@\n-file1 contents at rev 2\n+file1 contents in working tree\n\n")

    def test_diff_rev_tree_rev_tree(self):
        tree1 = self.b.repository.revision_tree(b'rev-2')
        tree2 = self.b.repository.revision_tree(b'rev-3')
        output = get_diff_as_string(tree1, tree2)
        self.assertEqualDiff(output, b"=== modified file 'file2'\n--- old/file2\t2006-04-01 00:00:00 +0000\n+++ new/file2\t2006-04-03 00:00:00 +0000\n@@ -1,1 +1,1 @@\n-file2 contents at rev 1\n+file2 contents at rev 3\n\n")

    def test_diff_add_files(self):
        tree1 = self.b.repository.revision_tree(_mod_revision.NULL_REVISION)
        tree2 = self.b.repository.revision_tree(b'rev-1')
        output = get_diff_as_string(tree1, tree2)
        self.assertEqualDiff(output, b"=== added file 'file1'\n--- old/file1\t1970-01-01 00:00:00 +0000\n+++ new/file1\t2006-04-01 00:00:00 +0000\n@@ -0,0 +1,1 @@\n+file1 contents at rev 1\n\n=== added file 'file2'\n--- old/file2\t1970-01-01 00:00:00 +0000\n+++ new/file2\t2006-04-01 00:00:00 +0000\n@@ -0,0 +1,1 @@\n+file2 contents at rev 1\n\n")

    def test_diff_remove_files(self):
        tree1 = self.b.repository.revision_tree(b'rev-3')
        tree2 = self.b.repository.revision_tree(b'rev-4')
        output = get_diff_as_string(tree1, tree2)
        self.assertEqualDiff(output, b"=== removed file 'file2'\n--- old/file2\t2006-04-03 00:00:00 +0000\n+++ new/file2\t1970-01-01 00:00:00 +0000\n@@ -1,1 +0,0 @@\n-file2 contents at rev 3\n\n")

    def test_show_diff_specified(self):
        """A working tree filename can be used to identify a file"""
        self.wt.rename_one('file1', 'file1b')
        old_tree = self.b.repository.revision_tree(b'rev-1')
        new_tree = self.b.repository.revision_tree(b'rev-4')
        out = get_diff_as_string(old_tree, new_tree, specific_files=['file1b'], working_tree=self.wt)
        self.assertContainsRe(out, b'file1\t')

    def test_recursive_diff(self):
        """Children of directories are matched"""
        os.mkdir('dir1')
        os.mkdir('dir2')
        self.wt.add(['dir1', 'dir2'])
        self.wt.rename_one('file1', 'dir1/file1')
        old_tree = self.b.repository.revision_tree(b'rev-1')
        new_tree = self.b.repository.revision_tree(b'rev-4')
        out = get_diff_as_string(old_tree, new_tree, specific_files=['dir1'], working_tree=self.wt)
        self.assertContainsRe(out, b'file1\t')
        out = get_diff_as_string(old_tree, new_tree, specific_files=['dir2'], working_tree=self.wt)
        self.assertNotContainsRe(out, b'file1\t')