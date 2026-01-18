from breezy import errors, osutils
from breezy.tests import features
from breezy.tests.per_lock import TestCaseWithLock
def _disabled_test_read_then_write_excludes(self):
    """If a file is read-locked, taking out a write lock should fail."""
    a_lock = self.read_lock('a-file')
    self.addCleanup(a_lock.unlock)
    self.assertRaises(errors.LockContention, self.write_lock, 'a-file')