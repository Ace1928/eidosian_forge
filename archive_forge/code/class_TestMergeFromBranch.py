import os
from breezy import conflicts, errors, merge
from breezy.tests import per_workingtree
from breezy.workingtree import PointlessMerge
class TestMergeFromBranch(per_workingtree.TestCaseWithWorkingTree):

    def create_two_trees_for_merging(self):
        """Create two trees that can be merged from.

        This sets self.tree_from, self.first_rev, self.tree_to, self.second_rev
        and self.to_second_rev.
        """
        self.tree_from = self.make_branch_and_tree('from')
        self.first_rev = self.tree_from.commit('first post')
        self.tree_to = self.tree_from.controldir.sprout('to').open_workingtree()
        self.second_rev = self.tree_from.commit('second rev on from', allow_pointless=True)
        self.to_second_rev = self.tree_to.commit('second rev on to', allow_pointless=True)

    def test_smoking_merge(self):
        """Smoke test of merge_from_branch."""
        self.create_two_trees_for_merging()
        self.tree_to.merge_from_branch(self.tree_from.branch)
        self.assertEqual([self.to_second_rev, self.second_rev], self.tree_to.get_parent_ids())

    def test_merge_to_revision(self):
        """Merge from a branch to a revision that is not the tip."""
        self.create_two_trees_for_merging()
        self.third_rev = self.tree_from.commit('real_tip')
        self.tree_to.merge_from_branch(self.tree_from.branch, to_revision=self.second_rev)
        self.assertEqual([self.to_second_rev, self.second_rev], self.tree_to.get_parent_ids())

    def test_compare_after_merge(self):
        tree_a = self.make_branch_and_tree('tree_a')
        self.build_tree_contents([('tree_a/file', b'text-a')])
        tree_a.add('file')
        tree_a.commit('added file')
        tree_b = tree_a.controldir.sprout('tree_b').open_workingtree()
        os.unlink('tree_a/file')
        tree_a.commit('deleted file')
        self.build_tree_contents([('tree_b/file', b'text-b')])
        tree_b.commit('changed file')
        tree_a.merge_from_branch(tree_b.branch)
        tree_a.lock_read()
        self.addCleanup(tree_a.unlock)
        list(tree_a.iter_changes(tree_a.basis_tree()))

    def test_merge_empty(self):
        tree_a = self.make_branch_and_tree('tree_a')
        self.build_tree_contents([('tree_a/file', b'text-a')])
        tree_a.add('file')
        tree_a.commit('added file')
        tree_b = self.make_branch_and_tree('treeb')
        self.assertRaises(errors.NoCommits, tree_a.merge_from_branch, tree_b.branch)
        tree_b.merge_from_branch(tree_a.branch)

    def test_merge_base(self):
        tree_a = self.make_branch_and_tree('tree_a')
        self.build_tree_contents([('tree_a/file', b'text-a')])
        tree_a.add('file')
        rev1 = tree_a.commit('added file')
        tree_b = tree_a.controldir.sprout('tree_b').open_workingtree()
        os.unlink('tree_a/file')
        tree_a.commit('deleted file')
        self.build_tree_contents([('tree_b/file', b'text-b')])
        tree_b.commit('changed file')
        self.assertRaises(PointlessMerge, tree_a.merge_from_branch, tree_b.branch, from_revision=tree_b.branch.last_revision())
        tree_a.merge_from_branch(tree_b.branch, from_revision=rev1)
        tree_a.lock_read()
        self.addCleanup(tree_a.unlock)
        changes = list(tree_a.iter_changes(tree_a.basis_tree()))
        self.assertEqual(1, len(changes), changes)

    def test_merge_type(self):
        this = self.make_branch_and_tree('this')
        self.build_tree_contents([('this/foo', b'foo')])
        this.add('foo')
        this.commit('added foo')
        other = this.controldir.sprout('other').open_workingtree()
        self.build_tree_contents([('other/foo', b'bar')])
        other.commit('content -> bar')
        self.build_tree_contents([('this/foo', b'baz')])
        this.commit('content -> baz')

        class QuxMerge(merge.Merge3Merger):

            def text_merge(self, trans_id, paths):
                self.tt.create_file([b'qux'], trans_id)
        this.merge_from_branch(other.branch, merge_type=QuxMerge)
        self.assertEqual(b'qux', this.get_file_text('foo'))