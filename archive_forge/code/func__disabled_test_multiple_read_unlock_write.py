from breezy import errors, osutils
from breezy.tests import features
from breezy.tests.per_lock import TestCaseWithLock
def _disabled_test_multiple_read_unlock_write(self):
    """We can only grab a write lock if all read locks are done."""
    a_lock = b_lock = c_lock = None
    try:
        a_lock = self.read_lock('a-file')
        b_lock = self.read_lock('a-file')
        self.assertRaises(errors.LockContention, self.write_lock, 'a-file')
        a_lock.unlock()
        a_lock = None
        self.assertRaises(errors.LockContention, self.write_lock, 'a-file')
        b_lock.unlock()
        b_lock = None
        c_lock = self.write_lock('a-file')
        c_lock.unlock()
        c_lock = None
    finally:
        if a_lock is not None:
            a_lock.unlock()
        if b_lock is not None:
            b_lock.unlock()
        if c_lock is not None:
            c_lock.unlock()