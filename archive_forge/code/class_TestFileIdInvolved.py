import sys
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy import tests, transform
from breezy.bzr import inventory, remote
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
class TestFileIdInvolved(FileIdInvolvedBase):
    scenarios = all_repository_vf_format_scenarios()

    def setUp(self):
        super().setUp()
        main_wt = self.make_branch_and_tree('main')
        main_branch = main_wt.branch
        self.build_tree(['main/a', 'main/b', 'main/c'])
        main_wt.add(['a', 'b', 'c'], ids=[b'a-file-id-2006-01-01-abcd', b'b-file-id-2006-01-01-defg', b'c-funky<file-id>quiji%bo'])
        try:
            main_wt.commit('Commit one', rev_id=b'rev-A')
        except errors.IllegalPath:
            if sys.platform == 'win32':
                raise tests.TestSkipped('Old repository formats do not support file ids with <> on win32')
            raise
        bt1 = self.make_branch_and_tree('branch1')
        bt1.pull(main_branch)
        b1 = bt1.branch
        self.build_tree(['branch1/d'])
        bt1.add(['d'], ids=[b'file-d'])
        bt1.commit('branch1, Commit one', rev_id=b'rev-E')
        self.touch(main_wt, 'a')
        main_wt.commit('Commit two', rev_id=b'rev-B')
        bt2 = self.make_branch_and_tree('branch2')
        bt2.pull(main_branch)
        branch2_branch = bt2.branch
        set_executability(bt2, 'b', True)
        bt2.commit('branch2, Commit one', rev_id=b'rev-J')
        main_wt.merge_from_branch(b1)
        main_wt.commit('merge branch1, rev-11', rev_id=b'rev-C')
        bt1.rename_one('d', 'e')
        bt1.commit('branch1, commit two', rev_id=b'rev-F')
        self.touch(bt2, 'c')
        bt2.commit('branch2, commit two', rev_id=b'rev-K')
        main_wt.merge_from_branch(b1)
        self.touch(main_wt, 'b')
        main_wt.commit('merge branch1, rev-12', rev_id=b'rev-<D>')
        main_wt.merge_from_branch(branch2_branch)
        main_wt.commit('merge branch1, rev-22', rev_id=b'rev-G')
        self.branch = main_branch

    def test_fileids_altered_between_two_revs(self):
        self.branch.lock_read()
        self.addCleanup(self.branch.unlock)
        self.branch.repository.fileids_altered_by_revision_ids([b'rev-J', b'rev-K'])
        self.assertEqual({b'b-file-id-2006-01-01-defg': {b'rev-J'}, b'c-funky<file-id>quiji%bo': {b'rev-K'}}, self.branch.repository.fileids_altered_by_revision_ids([b'rev-J', b'rev-K']))
        self.assertEqual({b'b-file-id-2006-01-01-defg': {b'rev-<D>'}, b'file-d': {b'rev-F'}}, self.branch.repository.fileids_altered_by_revision_ids([b'rev-<D>', b'rev-F']))
        self.assertEqual({b'b-file-id-2006-01-01-defg': {b'rev-<D>', b'rev-G', b'rev-J'}, b'c-funky<file-id>quiji%bo': {b'rev-K'}, b'file-d': {b'rev-F'}}, self.branch.repository.fileids_altered_by_revision_ids([b'rev-<D>', b'rev-G', b'rev-F', b'rev-K', b'rev-J']))
        self.assertEqual({b'a-file-id-2006-01-01-abcd': {b'rev-B'}, b'b-file-id-2006-01-01-defg': {b'rev-<D>', b'rev-G', b'rev-J'}, b'c-funky<file-id>quiji%bo': {b'rev-K'}, b'file-d': {b'rev-F'}}, self.branch.repository.fileids_altered_by_revision_ids([b'rev-G', b'rev-F', b'rev-C', b'rev-B', b'rev-<D>', b'rev-K', b'rev-J']))

    def fileids_altered_by_revision_ids(self, revision_ids):
        """This is a wrapper to strip TREE_ROOT if it occurs"""
        repo = self.branch.repository
        root_id = self.branch.basis_tree().path2id('')
        result = repo.fileids_altered_by_revision_ids(revision_ids)
        if root_id in result:
            del result[root_id]
        return result

    def test_fileids_altered_by_revision_ids(self):
        self.branch.lock_read()
        self.addCleanup(self.branch.unlock)
        self.assertEqual({b'a-file-id-2006-01-01-abcd': {b'rev-A'}, b'b-file-id-2006-01-01-defg': {b'rev-A'}, b'c-funky<file-id>quiji%bo': {b'rev-A'}}, self.fileids_altered_by_revision_ids([b'rev-A']))
        self.assertEqual({b'a-file-id-2006-01-01-abcd': {b'rev-B'}}, self.branch.repository.fileids_altered_by_revision_ids([b'rev-B']))
        self.assertEqual({b'b-file-id-2006-01-01-defg': {b'rev-<D>'}}, self.branch.repository.fileids_altered_by_revision_ids([b'rev-<D>']))

    def test_fileids_involved_full_compare(self):
        self.branch.lock_read()
        self.addCleanup(self.branch.unlock)
        pp = []
        graph = self.branch.repository.get_graph()
        history = list(graph.iter_lefthand_ancestry(self.branch.last_revision(), [_mod_revision.NULL_REVISION]))
        history.reverse()
        if len(history) < 2:
            return
        for start in range(0, len(history) - 1):
            start_id = history[start]
            for end in range(start + 1, len(history)):
                end_id = history[end]
                unique_revs = graph.find_unique_ancestors(end_id, [start_id])
                l1 = self.branch.repository.fileids_altered_by_revision_ids(unique_revs)
                l1 = set(l1.keys())
                l2 = self.compare_tree_fileids(self.branch, start_id, end_id)
                self.assertEqual(l1, l2)