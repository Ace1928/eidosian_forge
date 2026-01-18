import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
class TestDiff(DiffBase):

    def test_diff(self):
        tree = self.make_example_branch()
        self.build_tree_contents([('hello', b'hello world!')])
        tree.commit(message='fixing hello')
        output = self.run_bzr('diff -r 2..3', retcode=1)[0]
        self.assertTrue('\n+hello world!' in output)
        output = self.run_bzr('diff -c 3', retcode=1)[0]
        self.assertTrue('\n+hello world!' in output)
        output = self.run_bzr('diff -r last:3..last:1', retcode=1)[0]
        self.assertTrue('\n+baz' in output)
        output = self.run_bzr('diff -c last:2', retcode=1)[0]
        self.assertTrue('\n+baz' in output)
        self.build_tree(['moo'])
        tree.add('moo')
        os.unlink('moo')
        self.run_bzr('diff')

    def test_diff_prefix(self):
        """diff --prefix appends to filenames in output"""
        self.make_example_branch()
        self.build_tree_contents([('hello', b'hello world!\n')])
        out, err = self.run_bzr('diff --prefix old/:new/', retcode=1)
        self.assertEqual(err, '')
        self.assertEqualDiff(subst_dates(out), "=== modified file 'hello'\n--- old/hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n+++ new/hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n@@ -1,1 +1,1 @@\n-foo\n+hello world!\n\n")

    def test_diff_illegal_prefix_value(self):
        out, err = self.run_bzr('diff --prefix old/', retcode=3)
        self.assertContainsRe(err, '--prefix expects two values separated by a colon')

    def test_diff_p1(self):
        """diff -p1 produces lkml-style diffs"""
        self.make_example_branch()
        self.build_tree_contents([('hello', b'hello world!\n')])
        out, err = self.run_bzr('diff -p1', retcode=1)
        self.assertEqual(err, '')
        self.assertEqualDiff(subst_dates(out), "=== modified file 'hello'\n--- old/hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n+++ new/hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n@@ -1,1 +1,1 @@\n-foo\n+hello world!\n\n")

    def test_diff_p0(self):
        """diff -p0 produces diffs with no prefix"""
        self.make_example_branch()
        self.build_tree_contents([('hello', b'hello world!\n')])
        out, err = self.run_bzr('diff -p0', retcode=1)
        self.assertEqual(err, '')
        self.assertEqualDiff(subst_dates(out), "=== modified file 'hello'\n--- hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n+++ hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n@@ -1,1 +1,1 @@\n-foo\n+hello world!\n\n")

    def test_diff_nonexistent(self):
        self.make_example_branch()
        out, err = self.run_bzr('diff does-not-exist', retcode=3, error_regexes=('not versioned.*does-not-exist',))

    def test_diff_illegal_revision_specifiers(self):
        out, err = self.run_bzr('diff -r 1..23..123', retcode=3, error_regexes=('one or two revision specifiers',))

    def test_diff_using_and_format(self):
        out, err = self.run_bzr('diff --format=default --using=mydi', retcode=3, error_regexes=('are mutually exclusive',))

    def test_diff_nonexistent_revision(self):
        out, err = self.run_bzr('diff -r 123', retcode=3, error_regexes=("Requested revision: '123' does not exist in branch:",))

    def test_diff_nonexistent_dotted_revision(self):
        out, err = self.run_bzr('diff -r 1.1', retcode=3)
        self.assertContainsRe(err, "Requested revision: '1.1' does not exist in branch:")

    def test_diff_nonexistent_dotted_revision_change(self):
        out, err = self.run_bzr('diff -c 1.1', retcode=3)
        self.assertContainsRe(err, "Requested revision: '1.1' does not exist in branch:")

    def test_diff_unversioned(self):
        self.make_example_branch()
        self.build_tree(['unversioned-file'])
        out, err = self.run_bzr('diff unversioned-file', retcode=3)
        self.assertContainsRe(err, 'not versioned.*unversioned-file')

    def example_branches(self):
        branch1_tree = self.make_branch_and_tree('branch1')
        self.build_tree(['branch1/file'], line_endings='binary')
        self.build_tree(['branch1/file2'], line_endings='binary')
        branch1_tree.add('file')
        branch1_tree.add('file2')
        branch1_tree.commit(message='add file and file2')
        branch2_tree = branch1_tree.controldir.sprout('branch2').open_workingtree()
        self.build_tree_contents([('branch2/file', b'new content\n')])
        branch2_tree.commit(message='update file')
        return (branch1_tree, branch2_tree)

    def check_b2_vs_b1(self, cmd):
        out, err = self.run_bzr(cmd, retcode=1)
        self.assertEqual('', err)
        self.assertEqual("=== modified file 'file'\n--- old/file\tYYYY-MM-DD HH:MM:SS +ZZZZ\n+++ new/file\tYYYY-MM-DD HH:MM:SS +ZZZZ\n@@ -1,1 +1,1 @@\n-new content\n+contents of branch1/file\n\n", subst_dates(out))

    def check_b1_vs_b2(self, cmd):
        out, err = self.run_bzr(cmd, retcode=1)
        self.assertEqual('', err)
        self.assertEqualDiff("=== modified file 'file'\n--- old/file\tYYYY-MM-DD HH:MM:SS +ZZZZ\n+++ new/file\tYYYY-MM-DD HH:MM:SS +ZZZZ\n@@ -1,1 +1,1 @@\n-contents of branch1/file\n+new content\n\n", subst_dates(out))

    def check_no_diffs(self, cmd):
        out, err = self.run_bzr(cmd, retcode=0)
        self.assertEqual('', err)
        self.assertEqual('', out)

    def test_diff_branches(self):
        self.example_branches()
        self.check_b2_vs_b1('diff -r branch:branch2 branch1')
        self.check_b2_vs_b1('diff --old branch2 --new branch1')
        self.check_b2_vs_b1('diff --old branch2 branch1')
        self.check_b2_vs_b1('diff branch2 --new branch1')
        self.check_b2_vs_b1('diff --old branch2 --new branch1 file')
        self.check_b2_vs_b1('diff --old branch2 branch1/file')
        self.check_b2_vs_b1('diff branch2/file --new branch1')
        self.check_no_diffs('diff --old branch2 --new branch1 file2')
        self.check_no_diffs('diff --old branch2 branch1/file2')
        self.check_no_diffs('diff branch2/file2 --new branch1')

    def test_diff_branches_no_working_trees(self):
        branch1_tree, branch2_tree = self.example_branches()
        dir1 = branch1_tree.controldir
        dir1.destroy_workingtree()
        self.assertFalse(dir1.has_workingtree())
        self.check_b2_vs_b1('diff --old branch2 --new branch1')
        self.check_b2_vs_b1('diff --old branch2 branch1')
        self.check_b2_vs_b1('diff branch2 --new branch1')
        self.check_b1_vs_b2('diff --old branch1 --new branch2')
        self.check_b1_vs_b2('diff --old branch1 branch2')
        self.check_b1_vs_b2('diff branch1 --new branch2')
        dir2 = branch2_tree.controldir
        dir2.destroy_workingtree()
        self.assertFalse(dir2.has_workingtree())
        self.check_b1_vs_b2('diff --old branch1 --new branch2')
        self.check_b1_vs_b2('diff --old branch1 branch2')
        self.check_b1_vs_b2('diff branch1 --new branch2')

    def test_diff_revno_branches(self):
        self.example_branches()
        branch2_tree = workingtree.WorkingTree.open_containing('branch2')[0]
        self.build_tree_contents([('branch2/file', b'even newer content')])
        branch2_tree.commit(message='update file once more')
        out, err = self.run_bzr('diff -r revno:1:branch2..revno:1:branch1')
        self.assertEqual('', err)
        self.assertEqual('', out)
        out, err = self.run_bzr('diff -r revno:2:branch2..revno:1:branch1', retcode=1)
        self.assertEqual('', err)
        self.assertEqualDiff("=== modified file 'file'\n--- old/file\tYYYY-MM-DD HH:MM:SS +ZZZZ\n+++ new/file\tYYYY-MM-DD HH:MM:SS +ZZZZ\n@@ -1,1 +1,1 @@\n-new content\n+contents of branch1/file\n\n", subst_dates(out))

    def test_diff_color_always(self):
        from ... import colordiff
        from ...terminal import colorstring
        self.overrideAttr(colordiff, 'GLOBAL_COLORDIFFRC', None)
        self.example_branches()
        branch2_tree = workingtree.WorkingTree.open_containing('branch2')[0]
        self.build_tree_contents([('branch2/file', b'even newer content')])
        branch2_tree.commit(message='update file once more')
        out, err = self.run_bzr('diff --color=always -r revno:2:branch2..revno:1:branch1', retcode=1)
        self.assertEqual('', err)
        self.assertEqualDiff((colorstring(b"=== modified file 'file'\n", 'darkyellow') + colorstring(b'--- old/file\tYYYY-MM-DD HH:MM:SS +ZZZZ\n', 'darkred') + colorstring(b'+++ new/file\tYYYY-MM-DD HH:MM:SS +ZZZZ\n', 'darkblue') + colorstring(b'@@ -1 +1 @@\n', 'darkgreen') + colorstring(b'-new content\n', 'darkred') + colorstring(b'+contents of branch1/file\n', 'darkblue') + colorstring(b'\n', 'darkwhite')).decode(), subst_dates(out))

    def example_branch2(self):
        branch1_tree = self.make_branch_and_tree('branch1')
        self.build_tree_contents([('branch1/file1', b'original line\n')])
        branch1_tree.add('file1')
        branch1_tree.commit(message='first commit')
        self.build_tree_contents([('branch1/file1', b'repo line\n')])
        branch1_tree.commit(message='second commit')
        return branch1_tree

    def test_diff_to_working_tree(self):
        self.example_branch2()
        self.build_tree_contents([('branch1/file1', b'new line')])
        output = self.run_bzr('diff -r 1.. branch1', retcode=1)
        self.assertContainsRe(output[0], '\n\\-original line\n\\+new line\n')

    def test_diff_to_working_tree_in_subdir(self):
        self.example_branch2()
        self.build_tree_contents([('branch1/file1', b'new line')])
        os.mkdir('branch1/dir1')
        output = self.run_bzr('diff -r 1..', retcode=1, working_dir='branch1/dir1')
        self.assertContainsRe(output[0], '\n\\-original line\n\\+new line\n')

    def test_diff_across_rename(self):
        """The working tree path should always be considered for diffing"""
        tree = self.make_example_branch()
        self.run_bzr('diff -r 0..1 hello', retcode=1)
        tree.rename_one('hello', 'hello1')
        self.run_bzr('diff hello1', retcode=1)
        self.run_bzr('diff -r 0..1 hello1', retcode=1)

    def test_diff_to_branch_no_working_tree(self):
        branch1_tree = self.example_branch2()
        dir1 = branch1_tree.controldir
        dir1.destroy_workingtree()
        self.assertFalse(dir1.has_workingtree())
        output = self.run_bzr('diff -r 1.. branch1', retcode=1)
        self.assertContainsRe(output[0], '\n\\-original line\n\\+repo line\n')

    def test_custom_format(self):

        class BooDiffTree(DiffTree):

            def show_diff(self, specific_files, extra_trees=None):
                self.to_file.write('BOO!\n')
                return super().show_diff(specific_files, extra_trees)
        diff_format_registry.register('boo', BooDiffTree, 'Scary diff format')
        self.addCleanup(diff_format_registry.remove, 'boo')
        self.make_example_branch()
        self.build_tree_contents([('hello', b'hello world!\n')])
        output = self.run_bzr('diff --format=boo', retcode=1)
        self.assertTrue('BOO!' in output[0])
        output = self.run_bzr('diff -Fboo', retcode=1)
        self.assertTrue('BOO!' in output[0])

    def test_binary_diff_remove(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('a', b'\x00' * 20)])
        tree.add(['a'])
        tree.commit('add binary file')
        os.unlink('a')
        output = self.run_bzr('diff', retcode=1)
        self.assertEqual("=== removed file 'a'\nBinary files old/a and new/a differ\n", output[0])

    def test_moved_away(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('a', 'asdf\n')])
        tree.add(['a'])
        tree.commit('add a')
        tree.rename_one('a', 'b')
        self.build_tree_contents([('a', 'qwer\n')])
        tree.add('a')
        output, error = self.run_bzr('diff -p0', retcode=1)
        self.assertEqualDiff("=== added file 'a'\n--- a\tYYYY-MM-DD HH:MM:SS +ZZZZ\n+++ a\tYYYY-MM-DD HH:MM:SS +ZZZZ\n@@ -0,0 +1,1 @@\n+qwer\n\n=== renamed file 'a' => 'b'\n", subst_dates(output))