from testtools.matchers import Equals, Matcher, Mismatch
from .. import osutils
from .. import revision as _mod_revision
from ..tree import InterTree, TreeChange
class _AncestryMismatch(Mismatch):
    """Ancestry matching mismatch."""

    def __init__(self, tip_revision, got, expected):
        self.tip_revision = tip_revision
        self.got = got
        self.expected = expected

    def describe(self):
        return 'mismatched ancestry for revision {!r} was {!r}, expected {!r}'.format(self.tip_revision, self.got, self.expected)