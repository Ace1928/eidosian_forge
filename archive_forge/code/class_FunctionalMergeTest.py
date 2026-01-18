import os
import sys
import tempfile
import breezy
from .. import controldir, errors, merge_directive, osutils
from ..bzr import generate_ids
from ..bzr.conflicts import ContentsConflict, PathConflict, TextConflict
from ..merge import Diff3Merger, Merge3Merger, Merger, WeaveMerger
from ..osutils import getcwd, pathjoin
from ..workingtree import WorkingTree
from . import TestCaseWithTransport, TestSkipped, features
class FunctionalMergeTest(TestCaseWithTransport):

    def test_trivial_star_merge(self):
        """Test that merges in a star shape Just Work."""
        self.build_tree(('original/', 'original/file1', 'original/file2'))
        tree = self.make_branch_and_tree('original')
        branch = tree.branch
        tree.smart_add(['original'])
        tree.commit('start branch.', verbose=False)
        self.build_tree(('mary/',))
        branch.controldir.clone('mary')
        with open('original/file1', 'w') as f:
            f.write('John\n')
        tree.commit('change file1')
        mary_tree = WorkingTree.open('mary')
        mary_branch = mary_tree.branch
        with open('mary/file2', 'w') as f:
            f.write('Mary\n')
        mary_tree.commit('change file2')
        base = [None, None]
        other = ('mary', -1)
        tree.merge_from_branch(mary_tree.branch)
        with open('original/file1') as f:
            self.assertEqual('John\n', f.read())
        with open('original/file2') as f:
            self.assertEqual('Mary\n', f.read())

    def test_conflicts(self):
        wta = self.make_branch_and_tree('a')
        self.build_tree_contents([('a/file', b'contents\n')])
        wta.add('file')
        wta.commit('base revision', allow_pointless=False)
        d_b = wta.branch.controldir.clone('b')
        self.build_tree_contents([('a/file', b'other contents\n')])
        wta.commit('other revision', allow_pointless=False)
        self.build_tree_contents([('b/file', b'this contents contents\n')])
        wtb = d_b.open_workingtree()
        wtb.commit('this revision', allow_pointless=False)
        self.assertEqual(1, len(wtb.merge_from_branch(wta.branch)))
        self.assertPathExists('b/file.THIS')
        self.assertPathExists('b/file.BASE')
        self.assertPathExists('b/file.OTHER')
        wtb.revert()
        self.assertEqual(1, len(wtb.merge_from_branch(wta.branch, merge_type=WeaveMerger)))
        self.assertPathExists('b/file')
        self.assertPathExists('b/file.THIS')
        self.assertPathExists('b/file.BASE')
        self.assertPathExists('b/file.OTHER')

    def test_weave_conflicts_not_in_base(self):
        builder = self.make_branch_builder('source')
        builder.start_series()
        a_id = builder.build_snapshot(None, [('add', ('', b'TREE_ROOT', 'directory', None))])
        b_id = builder.build_snapshot([a_id], [])
        c_id = builder.build_snapshot([a_id], [('add', ('foo', b'foo-id', 'file', b'orig\ncontents\n'))])
        d_id = builder.build_snapshot([b_id, c_id], [('add', ('foo', b'foo-id', 'file', b'orig\ncontents\nand D\n'))])
        e_id = builder.build_snapshot([c_id, b_id], [('modify', ('foo', b'orig\ncontents\nand E\n'))])
        builder.finish_series()
        tree = builder.get_branch().create_checkout('tree', lightweight=True)
        self.assertEqual(1, len(tree.merge_from_branch(tree.branch, to_revision=d_id, merge_type=WeaveMerger)))
        self.assertPathExists('tree/foo.THIS')
        self.assertPathExists('tree/foo.OTHER')
        self.expectFailure('fail to create .BASE in some criss-cross merges', self.assertPathExists, 'tree/foo.BASE')
        self.assertPathExists('tree/foo.BASE')

    def test_merge_unrelated(self):
        """Sucessfully merges unrelated branches with no common names"""
        wta = self.make_branch_and_tree('a')
        a = wta.branch
        with open('a/a_file', 'wb') as f:
            f.write(b'contents\n')
        wta.add('a_file')
        wta.commit('a_revision', allow_pointless=False)
        wtb = self.make_branch_and_tree('b')
        b = wtb.branch
        with open('b/b_file', 'wb') as f:
            f.write(b'contents\n')
        wtb.add('b_file')
        b_rev = wtb.commit('b_revision', allow_pointless=False)
        wta.merge_from_branch(wtb.branch, b_rev, b'null:')
        self.assertTrue(os.path.lexists('a/b_file'))
        self.assertEqual([b_rev], wta.get_parent_ids()[1:])

    def test_merge_unrelated_conflicting(self):
        """Sucessfully merges unrelated branches with common names"""
        wta = self.make_branch_and_tree('a')
        a = wta.branch
        with open('a/file', 'wb') as f:
            f.write(b'contents\n')
        wta.add('file')
        wta.commit('a_revision', allow_pointless=False)
        wtb = self.make_branch_and_tree('b')
        b = wtb.branch
        with open('b/file', 'wb') as f:
            f.write(b'contents\n')
        wtb.add('file')
        b_rev = wtb.commit('b_revision', allow_pointless=False)
        wta.merge_from_branch(wtb.branch, b_rev, b'null:')
        self.assertTrue(os.path.lexists('a/file'))
        self.assertTrue(os.path.lexists('a/file.moved'))
        self.assertEqual([b_rev], wta.get_parent_ids()[1:])

    def test_merge_deleted_conflicts(self):
        wta = self.make_branch_and_tree('a')
        with open('a/file', 'wb') as f:
            f.write(b'contents\n')
        wta.add('file')
        wta.commit('a_revision', allow_pointless=False)
        self.run_bzr('branch a b')
        os.remove('a/file')
        wta.commit('removed file', allow_pointless=False)
        with open('b/file', 'wb') as f:
            f.write(b'changed contents\n')
        wtb = WorkingTree.open('b')
        wtb.commit('changed file', allow_pointless=False)
        wtb.merge_from_branch(wta.branch, wta.branch.last_revision(), wta.branch.get_rev_id(1))
        self.assertFalse(os.path.lexists('b/file'))

    def test_merge_metadata_vs_deletion(self):
        """Conflict deletion vs metadata change"""
        a_wt = self.make_branch_and_tree('a')
        with open('a/file', 'wb') as f:
            f.write(b'contents\n')
        a_wt.add('file')
        a_wt.commit('r0')
        self.run_bzr('branch a b')
        b_wt = WorkingTree.open('b')
        os.chmod('b/file', 493)
        os.remove('a/file')
        a_wt.commit('removed a')
        self.assertEqual(a_wt.branch.revno(), 2)
        self.assertFalse(os.path.exists('a/file'))
        b_wt.commit('exec a')
        a_wt.merge_from_branch(b_wt.branch, b_wt.last_revision(), b'null:')
        self.assertTrue(os.path.exists('a/file'))

    def test_merge_swapping_renames(self):
        a_wt = self.make_branch_and_tree('a')
        with open('a/un', 'wb') as f:
            f.write(b'UN')
        with open('a/deux', 'wb') as f:
            f.write(b'DEUX')
        a_wt.add('un')
        a_wt.add('deux')
        a_wt.commit('r0', rev_id=b'r0')
        self.run_bzr('branch a b')
        b_wt = WorkingTree.open('b')
        b_wt.rename_one('un', 'tmp')
        b_wt.rename_one('deux', 'un')
        b_wt.rename_one('tmp', 'deux')
        b_wt.commit('r1', rev_id=b'r1')
        self.assertEqual(0, len(a_wt.merge_from_branch(b_wt.branch, b_wt.branch.last_revision(), b_wt.branch.get_rev_id(1))))
        self.assertPathExists('a/un')
        self.assertTrue('a/deux')
        self.assertFalse(os.path.exists('a/tmp'))
        with open('a/un') as f:
            self.assertEqual(f.read(), 'DEUX')
        with open('a/deux') as f:
            self.assertEqual(f.read(), 'UN')

    def test_merge_delete_and_add_same(self):
        a_wt = self.make_branch_and_tree('a')
        with open('a/file', 'wb') as f:
            f.write(b'THIS')
        a_wt.add('file')
        a_wt.commit('r0')
        self.run_bzr('branch a b')
        b_wt = WorkingTree.open('b')
        os.remove('b/file')
        b_wt.commit('r1')
        with open('b/file', 'wb') as f:
            f.write(b'THAT')
        b_wt.add('file')
        b_wt.commit('r2')
        a_wt.merge_from_branch(b_wt.branch, b_wt.branch.last_revision(), b_wt.branch.get_rev_id(1))
        self.assertTrue(os.path.exists('a/file'))
        with open('a/file') as f:
            self.assertEqual(f.read(), 'THAT')

    def test_merge_rename_before_create(self):
        """rename before create

        This case requires that you must not do creates
        before move-into-place:

        $ touch foo
        $ bzr add foo
        $ bzr commit
        $ bzr mv foo bar
        $ touch foo
        $ bzr add foo
        $ bzr commit
        """
        a_wt = self.make_branch_and_tree('a')
        with open('a/foo', 'wb') as f:
            f.write(b'A/FOO')
        a_wt.add('foo')
        a_wt.commit('added foo')
        self.run_bzr('branch a b')
        b_wt = WorkingTree.open('b')
        b_wt.rename_one('foo', 'bar')
        with open('b/foo', 'wb') as f:
            f.write(b'B/FOO')
        b_wt.add('foo')
        b_wt.commit('moved foo to bar, added new foo')
        a_wt.merge_from_branch(b_wt.branch, b_wt.branch.last_revision(), b_wt.branch.get_rev_id(1))

    def test_merge_create_before_rename(self):
        """create before rename, target parents before children

        This case requires that you must not do move-into-place
        before creates, and that you must not do children after
        parents:

        $ touch foo
        $ bzr add foo
        $ bzr commit
        $ bzr mkdir bar
        $ bzr add bar
        $ bzr mv foo bar/foo
        $ bzr commit
        """
        os.mkdir('a')
        a_wt = self.make_branch_and_tree('a')
        with open('a/foo', 'wb') as f:
            f.write(b'A/FOO')
        a_wt.add('foo')
        a_wt.commit('added foo')
        self.run_bzr('branch a b')
        b_wt = WorkingTree.open('b')
        os.mkdir('b/bar')
        b_wt.add('bar')
        b_wt.rename_one('foo', 'bar/foo')
        b_wt.commit('created bar dir, moved foo into bar')
        a_wt.merge_from_branch(b_wt.branch, b_wt.branch.last_revision(), b_wt.branch.get_rev_id(1))

    def test_merge_rename_to_temp_before_delete(self):
        """rename to temp before delete, source children before parents

        This case requires that you must not do deletes before
        move-out-of-the-way, and that you must not do children
        after parents:

        $ mkdir foo
        $ touch foo/bar
        $ bzr add foo/bar
        $ bzr commit
        $ bzr mv foo/bar bar
        $ rmdir foo
        $ bzr commit
        """
        a_wt = self.make_branch_and_tree('a')
        os.mkdir('a/foo')
        with open('a/foo/bar', 'wb') as f:
            f.write(b'A/FOO/BAR')
        a_wt.add('foo')
        a_wt.add('foo/bar')
        a_wt.commit('added foo/bar')
        self.run_bzr('branch a b')
        b_wt = WorkingTree.open('b')
        b_wt.rename_one('foo/bar', 'bar')
        os.rmdir('b/foo')
        b_wt.remove('foo')
        b_wt.commit('moved foo/bar to bar, deleted foo')
        a_wt.merge_from_branch(b_wt.branch, b_wt.branch.last_revision(), b_wt.branch.get_rev_id(1))

    def test_merge_delete_before_rename_to_temp(self):
        """delete before rename to temp

        This case requires that you must not do
        move-out-of-the-way before deletes:

        $ touch foo
        $ touch bar
        $ bzr add foo bar
        $ bzr commit
        $ rm foo
        $ bzr rm foo
        $ bzr mv bar foo
        $ bzr commit
        """
        a_wt = self.make_branch_and_tree('a')
        with open('a/foo', 'wb') as f:
            f.write(b'A/FOO')
        with open('a/bar', 'wb') as f:
            f.write(b'A/BAR')
        a_wt.add('foo')
        a_wt.add('bar')
        a_wt.commit('added foo and bar')
        self.run_bzr('branch a b')
        b_wt = WorkingTree.open('b')
        os.unlink('b/foo')
        b_wt.remove('foo')
        b_wt.rename_one('bar', 'foo')
        b_wt.commit('deleted foo, renamed bar to foo')
        a_wt.merge_from_branch(b_wt.branch, b_wt.branch.last_revision(), b_wt.branch.get_rev_id(1))