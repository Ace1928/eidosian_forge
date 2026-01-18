import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
class TestMergeImplementation(TestCaseWithTransport):

    def do_merge(self, target_tree, source_tree, **kwargs):
        merger = _mod_merge.Merger.from_revision_ids(target_tree, source_tree.last_revision(), other_branch=source_tree.branch)
        merger.merge_type = self.merge_type
        for name, value in kwargs.items():
            setattr(merger, name, value)
        merger.do_merge()

    def test_merge_specific_file(self):
        this_tree = self.make_branch_and_tree('this')
        this_tree.lock_write()
        self.addCleanup(this_tree.unlock)
        self.build_tree_contents([('this/file1', b'a\nb\n'), ('this/file2', b'a\nb\n')])
        this_tree.add(['file1', 'file2'])
        this_tree.commit('Added files')
        other_tree = this_tree.controldir.sprout('other').open_workingtree()
        self.build_tree_contents([('other/file1', b'a\nb\nc\n'), ('other/file2', b'a\nb\nc\n')])
        other_tree.commit('modified both')
        self.build_tree_contents([('this/file1', b'd\na\nb\n'), ('this/file2', b'd\na\nb\n')])
        this_tree.commit('modified both')
        self.do_merge(this_tree, other_tree, interesting_files=['file1'])
        self.assertFileEqual(b'd\na\nb\nc\n', 'this/file1')
        self.assertFileEqual(b'd\na\nb\n', 'this/file2')

    def test_merge_move_and_change(self):
        this_tree = self.make_branch_and_tree('this')
        this_tree.lock_write()
        self.addCleanup(this_tree.unlock)
        self.build_tree_contents([('this/file1', b'line 1\nline 2\nline 3\nline 4\n')])
        this_tree.add('file1')
        this_tree.commit('Added file')
        other_tree = this_tree.controldir.sprout('other').open_workingtree()
        self.build_tree_contents([('other/file1', b'line 1\nline 2 to 2.1\nline 3\nline 4\n')])
        other_tree.commit('Changed 2 to 2.1')
        self.build_tree_contents([('this/file1', b'line 1\nline 3\nline 2\nline 4\n')])
        this_tree.commit('Swapped 2 & 3')
        self.do_merge(this_tree, other_tree)
        if self.merge_type is _mod_merge.LCAMerger:
            self.expectFailure("lca merge doesn't conflict for move and change", self.assertFileEqual, 'line 1\n<<<<<<< TREE\nline 3\nline 2\n=======\nline 2 to 2.1\nline 3\n>>>>>>> MERGE-SOURCE\nline 4\n', 'this/file1')
        else:
            self.assertFileEqual('line 1\n<<<<<<< TREE\nline 3\nline 2\n=======\nline 2 to 2.1\nline 3\n>>>>>>> MERGE-SOURCE\nline 4\n', 'this/file1')

    def test_modify_conflicts_with_delete(self):
        builder = self.make_branch_builder('test')
        builder.start_series()
        base_id = builder.build_snapshot(None, [('add', ('', None, 'directory', None)), ('add', ('foo', b'foo-id', 'file', b'a\nb\nc\nd\ne\n'))])
        other_id = builder.build_snapshot([base_id], [('modify', ('foo', b'a\nc\nd\ne\n'))])
        this_id = builder.build_snapshot([base_id], [('modify', ('foo', b'a\nb2\nc\nd\nX\ne\n'))])
        builder.finish_series()
        branch = builder.get_branch()
        this_tree = branch.controldir.create_workingtree()
        this_tree.lock_write()
        self.addCleanup(this_tree.unlock)
        other_tree = this_tree.controldir.sprout('other', other_id).open_workingtree()
        self.do_merge(this_tree, other_tree)
        if self.merge_type is _mod_merge.LCAMerger:
            self.expectFailure("lca merge doesn't track deleted lines", self.assertFileEqual, 'a\n<<<<<<< TREE\nb2\n=======\n>>>>>>> MERGE-SOURCE\nc\nd\nX\ne\n', 'test/foo')
        else:
            self.assertFileEqual(b'a\n<<<<<<< TREE\nb2\n=======\n>>>>>>> MERGE-SOURCE\nc\nd\nX\ne\n', 'test/foo')

    def get_limbodir_deletiondir(self, wt):
        transform = wt.transform()
        limbodir = transform._limbodir
        deletiondir = transform._deletiondir
        transform.finalize()
        return (limbodir, deletiondir)

    def test_merge_with_existing_limbo_empty(self):
        """Empty limbo dir is just cleaned up - see bug 427773"""
        wt = self.make_branch_and_tree('this')
        limbodir, deletiondir = self.get_limbodir_deletiondir(wt)
        os.mkdir(limbodir)
        self.do_merge(wt, wt)

    def test_merge_with_existing_limbo_non_empty(self):
        wt = self.make_branch_and_tree('this')
        limbodir, deletiondir = self.get_limbodir_deletiondir(wt)
        os.mkdir(limbodir)
        os.mkdir(os.path.join(limbodir, 'something'))
        self.assertRaises(errors.ExistingLimbo, self.do_merge, wt, wt)
        self.assertRaises(errors.LockError, wt.unlock)

    def test_merge_with_pending_deletion_empty(self):
        wt = self.make_branch_and_tree('this')
        limbodir, deletiondir = self.get_limbodir_deletiondir(wt)
        os.mkdir(deletiondir)
        self.do_merge(wt, wt)

    def test_merge_with_pending_deletion_non_empty(self):
        """Also see bug 427773"""
        wt = self.make_branch_and_tree('this')
        limbodir, deletiondir = self.get_limbodir_deletiondir(wt)
        os.mkdir(deletiondir)
        os.mkdir(os.path.join(deletiondir, 'something'))
        self.assertRaises(errors.ExistingPendingDeletion, self.do_merge, wt, wt)
        self.assertRaises(errors.LockError, wt.unlock)