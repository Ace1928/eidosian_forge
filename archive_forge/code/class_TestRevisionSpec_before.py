import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
class TestRevisionSpec_before(TestRevisionSpec):

    def test_int(self):
        self.assertInHistoryIs(1, b'r1', 'before:2')
        self.assertInHistoryIs(1, b'r1', 'before:-1')

    def test_before_one(self):
        self.assertInHistoryIs(0, b'null:', 'before:1')

    def test_before_none(self):
        self.assertInvalid('before:0', extra='\ncannot go before the null: revision')

    def test_revid(self):
        self.assertInHistoryIs(1, b'r1', 'before:revid:r2')

    def test_last(self):
        self.assertInHistoryIs(1, b'r1', 'before:last:1')

    def test_alt_revid(self):
        self.assertInHistoryIs(1, b'r1', 'before:revid:alt_r2')

    def test_alt_no_parents(self):
        new_tree = self.make_branch_and_tree('new_tree')
        new_tree.commit('first', rev_id=b'new_r1')
        self.tree.branch.fetch(new_tree.branch, b'new_r1')
        self.assertInHistoryIs(0, b'null:', 'before:revid:new_r1')

    def test_as_revision_id(self):
        self.assertAsRevisionId(b'r1', 'before:revid:r2')
        self.assertAsRevisionId(b'r1', 'before:2')
        self.assertAsRevisionId(b'r1', 'before:1.1.1')
        self.assertAsRevisionId(b'r1', 'before:revid:alt_r2')