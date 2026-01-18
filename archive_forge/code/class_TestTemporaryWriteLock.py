from breezy import errors
from breezy.tests.per_lock import TestCaseWithLock
class TestTemporaryWriteLock(TestCaseWithLock):

    def setUp(self):
        super().setUp()
        self.build_tree(['a-file'])

    def test_can_upgrade_and_write(self):
        """With only one lock, we should be able to write lock and switch back."""
        a_lock = self.read_lock('a-file')
        try:
            success, t_write_lock = a_lock.temporary_write_lock()
            self.assertTrue(success, 'We failed to grab a write lock.')
            try:
                self.assertEqual(b'contents of a-file\n', t_write_lock.f.read())
                t_write_lock.f.seek(0)
                t_write_lock.f.write(b'new contents for a-file\n')
                t_write_lock.f.seek(0)
                self.assertEqual(b'new contents for a-file\n', t_write_lock.f.read())
            finally:
                a_lock = t_write_lock.restore_read_lock()
        finally:
            a_lock.unlock()

    def test_is_write_locked(self):
        """With a temporary write lock, we cannot grab another lock."""
        a_lock = self.read_lock('a-file')
        try:
            success, t_write_lock = a_lock.temporary_write_lock()
            self.assertTrue(success, 'We failed to grab a write lock.')
            try:
                self.assertRaises(errors.LockContention, self.write_lock, 'a-file')
            finally:
                a_lock = t_write_lock.restore_read_lock()
            b_lock = self.read_lock('a-file')
            b_lock.unlock()
        finally:
            a_lock.unlock()

    def test_fails_when_locked(self):
        """We can't upgrade to a write lock if something else locks."""
        a_lock = self.read_lock('a-file')
        try:
            b_lock = self.read_lock('a-file')
            try:
                success, alt_lock = a_lock.temporary_write_lock()
                self.assertFalse(success)
                self.assertTrue(alt_lock is a_lock or a_lock.f is None)
                a_lock = alt_lock
                c_lock = self.read_lock('a-file')
                c_lock.unlock()
            finally:
                b_lock.unlock()
        finally:
            a_lock.unlock()