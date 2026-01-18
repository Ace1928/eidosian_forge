import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
class RevisionSpec_bork(RevisionSpec):
    prefix = 'irrelevant:'

    def _match_on(self, branch, revs):
        if self.spec == 'bork':
            return RevisionInfo.from_revision_id(branch, b'r1')
        else:
            raise InvalidRevisionSpec(self.spec, branch)