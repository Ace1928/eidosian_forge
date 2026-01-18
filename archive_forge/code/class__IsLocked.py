from testtools.matchers import Equals, Matcher, Mismatch
from .. import osutils
from .. import revision as _mod_revision
from ..tree import InterTree, TreeChange
class _IsLocked(Mismatch):
    """Something is locked."""

    def __init__(self, lockable_thing):
        self.lockable_thing = lockable_thing

    def describe(self):
        return '%s is locked' % self.lockable_thing