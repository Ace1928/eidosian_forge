from .. import debug, errors, lock, tests
from .scenarios import load_tests_apply_scenarios
class TestOSLock(tests.TestCaseInTempDir):
    scenarios = [(name, {'write_lock': write_lock, 'read_lock': read_lock}) for name, write_lock, read_lock in lock._lock_classes]
    read_lock = None
    write_lock = None

    def setUp(self):
        super().setUp()
        self.build_tree(['a-lock-file'])

    def test_create_read_lock(self):
        r_lock = self.read_lock('a-lock-file')
        r_lock.unlock()

    def test_create_write_lock(self):
        w_lock = self.write_lock('a-lock-file')
        w_lock.unlock()

    def test_read_locks_share(self):
        r_lock = self.read_lock('a-lock-file')
        try:
            lock2 = self.read_lock('a-lock-file')
            lock2.unlock()
        finally:
            r_lock.unlock()

    def test_write_locks_are_exclusive(self):
        w_lock = self.write_lock('a-lock-file')
        try:
            self.assertRaises(errors.LockContention, self.write_lock, 'a-lock-file')
        finally:
            w_lock.unlock()

    def test_read_locks_block_write_locks(self):
        r_lock = self.read_lock('a-lock-file')
        try:
            if lock.have_fcntl and self.write_lock is lock._fcntl_WriteLock:
                debug.debug_flags.add('strict_locks')
                self.assertRaises(errors.LockContention, self.write_lock, 'a-lock-file')
                debug.debug_flags.remove('strict_locks')
                try:
                    w_lock = self.write_lock('a-lock-file')
                except errors.LockContention:
                    self.fail('Unexpected success. fcntl read locks do not usually block write locks')
                else:
                    w_lock.unlock()
                    self.knownFailure("fcntl read locks don't block write locks without -Dlock")
            else:
                self.assertRaises(errors.LockContention, self.write_lock, 'a-lock-file')
        finally:
            r_lock.unlock()

    def test_write_locks_block_read_lock(self):
        w_lock = self.write_lock('a-lock-file')
        try:
            if lock.have_fcntl and self.read_lock is lock._fcntl_ReadLock:
                debug.debug_flags.add('strict_locks')
                self.assertRaises(errors.LockContention, self.read_lock, 'a-lock-file')
                debug.debug_flags.remove('strict_locks')
                try:
                    r_lock = self.read_lock('a-lock-file')
                except errors.LockContention:
                    self.fail('Unexpected success. fcntl write locks do not usually block read locks')
                else:
                    r_lock.unlock()
                    self.knownFailure("fcntl write locks don't block read locks without -Dlock")
            else:
                self.assertRaises(errors.LockContention, self.read_lock, 'a-lock-file')
        finally:
            w_lock.unlock()

    def test_temporary_write_lock(self):
        r_lock = self.read_lock('a-lock-file')
        try:
            status, w_lock = r_lock.temporary_write_lock()
            self.assertTrue(status)
            try:
                self.assertRaises(errors.LockContention, self.write_lock, 'a-lock-file')
            finally:
                r_lock = w_lock.restore_read_lock()
            r_lock2 = self.read_lock('a-lock-file')
            r_lock2.unlock()
        finally:
            r_lock.unlock()

    def test_temporary_write_lock_fails(self):
        r_lock = self.read_lock('a-lock-file')
        try:
            r_lock2 = self.read_lock('a-lock-file')
            try:
                status, w_lock = r_lock.temporary_write_lock()
                self.assertFalse(status)
                r_lock = w_lock
            finally:
                r_lock2.unlock()
            r_lock2 = self.read_lock('a-lock-file')
            r_lock2.unlock()
        finally:
            r_lock.unlock()