import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
class TestRevisionSpec_branch(TestRevisionSpec):

    def test_non_exact_branch(self):
        self.assertRaises(errors.NotBranchError, self.get_in_history, 'branch:tree2/a')

    def test_simple(self):
        self.assertInHistoryIs(None, b'alt_r2', 'branch:tree2')

    def test_self(self):
        self.assertInHistoryIs(2, b'r2', 'branch:tree')

    def test_unrelated(self):
        new_tree = self.make_branch_and_tree('new_tree')
        new_tree.commit('Commit one', rev_id=b'new_r1')
        new_tree.commit('Commit two', rev_id=b'new_r2')
        new_tree.commit('Commit three', rev_id=b'new_r3')
        self.assertInHistoryIs(None, b'new_r3', 'branch:new_tree')
        self.assertTrue(self.tree.branch.repository.has_revision(b'new_r3'))

    def test_no_commits(self):
        self.make_branch_and_tree('new_tree')
        self.assertRaises(errors.NoCommits, self.get_in_history, 'branch:new_tree')
        self.assertRaises(errors.NoCommits, self.get_as_tree, 'branch:new_tree')

    def test_as_revision_id(self):
        self.assertAsRevisionId(b'alt_r2', 'branch:tree2')

    def test_as_tree(self):
        tree = self.get_as_tree('branch:tree', self.tree2)
        self.assertEqual(b'r2', tree.get_revision_id())
        self.assertFalse(self.tree2.branch.repository.has_revision(b'r2'))