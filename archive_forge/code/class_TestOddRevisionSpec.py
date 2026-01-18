import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
class TestOddRevisionSpec(TestRevisionSpec):
    """Test things that aren't normally thought of as revision specs"""

    def test_none(self):
        self.assertInHistoryIs(None, None, None)

    def test_object(self):
        self.assertRaises(TypeError, RevisionSpec.from_string, object())