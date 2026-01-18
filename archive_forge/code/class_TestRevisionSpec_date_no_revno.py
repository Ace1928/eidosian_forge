import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
class TestRevisionSpec_date_no_revno(TestRevisionSpec_date):

    def get_in_history(self, revision_spec):
        old_revno = self.overrideAttr(self.tree.branch, 'revno', lambda: None)
        try:
            return spec_in_history(revision_spec, self.tree.branch)
        finally:
            self.tree.branch.revno = old_revno

    def test_today(self):
        self.assertInHistoryIs(2, self.revid2, 'date:today')