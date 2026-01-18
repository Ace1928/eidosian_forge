from breezy import errors, osutils
from breezy.tests import features
from breezy.tests.per_lock import TestCaseWithLock
def _disabled_test_write_then_read_excludes(self):
    """If a file is write-locked, taking out a read lock should fail.

        The file is exclusively owned by the write lock, so we shouldn't be
        able to take out a shared read lock.
        """
    a_lock = self.write_lock('a-file')
    self.addCleanup(a_lock.unlock)
    self.assertRaises(errors.LockContention, self.read_lock, 'a-file')