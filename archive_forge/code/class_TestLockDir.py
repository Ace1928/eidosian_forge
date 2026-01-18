import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
class TestLockDir(TestCaseWithTransport):
    """Test LockDir operations"""

    def logging_report_function(self, fmt, *args):
        self._logged_reports.append((fmt, args))

    def setup_log_reporter(self, lock_dir):
        self._logged_reports = []
        lock_dir._report_function = self.logging_report_function

    def test_00_lock_creation(self):
        """Creation of lock file on a transport"""
        t = self.get_transport()
        lf = LockDir(t, 'test_lock')
        self.assertFalse(lf.is_held)

    def test_01_lock_repr(self):
        """Lock string representation"""
        lf = LockDir(self.get_transport(), 'test_lock')
        r = repr(lf)
        self.assertContainsRe(r, '^LockDir\\(.*/test_lock\\)$')

    def test_02_unlocked_peek(self):
        lf = LockDir(self.get_transport(), 'test_lock')
        self.assertEqual(lf.peek(), None)

    def get_lock(self):
        return LockDir(self.get_transport(), 'test_lock')

    def test_unlock_after_break_raises(self):
        ld = self.get_lock()
        ld2 = self.get_lock()
        ld.create()
        ld.attempt_lock()
        ld2.force_break(ld2.peek())
        self.assertRaises(LockBroken, ld.unlock)

    def test_03_readonly_peek(self):
        lf = LockDir(self.get_readonly_transport(), 'test_lock')
        self.assertEqual(lf.peek(), None)

    def test_10_lock_uncontested(self):
        """Acquire and release a lock"""
        t = self.get_transport()
        lf = LockDir(t, 'test_lock')
        lf.create()
        lf.attempt_lock()
        try:
            self.assertTrue(lf.is_held)
        finally:
            lf.unlock()
            self.assertFalse(lf.is_held)

    def test_11_create_readonly_transport(self):
        """Fail to create lock on readonly transport"""
        t = self.get_readonly_transport()
        lf = LockDir(t, 'test_lock')
        self.assertRaises(LockFailed, lf.create)

    def test_12_lock_readonly_transport(self):
        """Fail to lock on readonly transport"""
        lf = LockDir(self.get_transport(), 'test_lock')
        lf.create()
        lf = LockDir(self.get_readonly_transport(), 'test_lock')
        self.assertRaises(LockFailed, lf.attempt_lock)

    def test_20_lock_contested(self):
        """Contention to get a lock"""
        t = self.get_transport()
        lf1 = LockDir(t, 'test_lock')
        lf1.create()
        lf1.attempt_lock()
        lf2 = LockDir(t, 'test_lock')
        try:
            lf2.attempt_lock()
            self.fail('Failed to detect lock collision')
        except LockContention as e:
            self.assertEqual(e.lock, lf2)
            self.assertContainsRe(str(e), '^Could not acquire.*test_lock.*$')
        lf1.unlock()

    def test_20_lock_peek(self):
        """Peek at the state of a lock"""
        t = self.get_transport()
        lf1 = LockDir(t, 'test_lock')
        lf1.create()
        lf1.attempt_lock()
        self.addCleanup(lf1.unlock)
        info1 = lf1.peek()
        self.assertEqual(set(info1.info_dict.keys()), {'user', 'nonce', 'hostname', 'pid', 'start_time'})
        info2 = LockDir(t, 'test_lock').peek()
        self.assertEqual(info1, info2)
        self.assertEqual(LockDir(t, 'other_lock').peek(), None)

    def test_21_peek_readonly(self):
        """Peek over a readonly transport"""
        t = self.get_transport()
        lf1 = LockDir(t, 'test_lock')
        lf1.create()
        lf2 = LockDir(self.get_readonly_transport(), 'test_lock')
        self.assertEqual(lf2.peek(), None)
        lf1.attempt_lock()
        self.addCleanup(lf1.unlock)
        info2 = lf2.peek()
        self.assertTrue(info2)
        self.assertEqual(info2.nonce, lf1.nonce)

    def test_30_lock_wait_fail(self):
        """Wait on a lock, then fail

        We ask to wait up to 400ms; this should fail within at most one
        second.  (Longer times are more realistic but we don't want the test
        suite to take too long, and this should do for now.)
        """
        t = self.get_transport()
        lf1 = LockDir(t, 'test_lock')
        lf1.create()
        lf2 = LockDir(t, 'test_lock')
        self.setup_log_reporter(lf2)
        lf1.attempt_lock()
        try:
            before = time.time()
            self.assertRaises(LockContention, lf2.wait_lock, timeout=0.4, poll=0.1)
            after = time.time()
            self.assertTrue(after - before <= 8.0, 'took %f seconds to detect lock contention' % (after - before))
        finally:
            lf1.unlock()
        self.assertEqual(1, len(self._logged_reports))
        self.assertContainsRe(self._logged_reports[0][0], 'Unable to obtain lock .* held by jrandom@example\\.com on .* \\(process #\\d+\\), acquired .* ago\\.\\nWill continue to try until \\d{2}:\\d{2}:\\d{2}, unless you press Ctrl-C.\\nSee "brz help break-lock" for more.')

    def test_31_lock_wait_easy(self):
        """Succeed when waiting on a lock with no contention.
        """
        t = self.get_transport()
        lf1 = LockDir(t, 'test_lock')
        lf1.create()
        self.setup_log_reporter(lf1)
        try:
            before = time.time()
            lf1.wait_lock(timeout=0.4, poll=0.1)
            after = time.time()
            self.assertTrue(after - before <= 1.0)
        finally:
            lf1.unlock()
        self.assertEqual([], self._logged_reports)

    def test_40_confirm_easy(self):
        """Confirm a lock that's already held"""
        t = self.get_transport()
        lf1 = LockDir(t, 'test_lock')
        lf1.create()
        lf1.attempt_lock()
        self.addCleanup(lf1.unlock)
        lf1.confirm()

    def test_41_confirm_not_held(self):
        """Confirm a lock that's already held"""
        t = self.get_transport()
        lf1 = LockDir(t, 'test_lock')
        lf1.create()
        self.assertRaises(LockNotHeld, lf1.confirm)

    def test_42_confirm_broken_manually(self):
        """Confirm a lock broken by hand"""
        t = self.get_transport()
        lf1 = LockDir(t, 'test_lock')
        lf1.create()
        lf1.attempt_lock()
        t.move('test_lock', 'lock_gone_now')
        self.assertRaises(LockBroken, lf1.confirm)
        t.move('lock_gone_now', 'test_lock')
        lf1.unlock()

    def test_43_break(self):
        """Break a lock whose caller has forgotten it"""
        t = self.get_transport()
        lf1 = LockDir(t, 'test_lock')
        lf1.create()
        lf1.attempt_lock()
        del lf1
        lf2 = LockDir(t, 'test_lock')
        holder_info = lf2.peek()
        self.assertTrue(holder_info)
        lf2.force_break(holder_info)
        lf2.attempt_lock()
        self.addCleanup(lf2.unlock)
        lf2.confirm()

    def test_44_break_already_released(self):
        """Lock break races with regular release"""
        t = self.get_transport()
        lf1 = LockDir(t, 'test_lock')
        lf1.create()
        lf1.attempt_lock()
        lf2 = LockDir(t, 'test_lock')
        holder_info = lf2.peek()
        lf1.unlock()
        lf2.force_break(holder_info)
        lf2.attempt_lock()
        self.addCleanup(lf2.unlock)
        lf2.confirm()

    def test_45_break_mismatch(self):
        """Lock break races with someone else acquiring it"""
        t = self.get_transport()
        lf1 = LockDir(t, 'test_lock')
        lf1.create()
        lf1.attempt_lock()
        lf2 = LockDir(t, 'test_lock')
        holder_info = lf2.peek()
        lf1.unlock()
        lf3 = LockDir(t, 'test_lock')
        lf3.attempt_lock()
        self.assertRaises(LockBreakMismatch, lf2.force_break, holder_info)
        lf3.unlock()

    def test_46_fake_read_lock(self):
        t = self.get_transport()
        lf1 = LockDir(t, 'test_lock')
        lf1.create()
        lf1.lock_read()
        lf1.unlock()

    def test_50_lockdir_representation(self):
        """Check the on-disk representation of LockDirs is as expected.

        There should always be a top-level directory named by the lock.
        When the lock is held, there should be a lockname/held directory
        containing an info file.
        """
        t = self.get_transport()
        lf1 = LockDir(t, 'test_lock')
        lf1.create()
        self.assertTrue(t.has('test_lock'))
        lf1.lock_write()
        self.assertTrue(t.has('test_lock/held/info'))
        lf1.unlock()
        self.assertFalse(t.has('test_lock/held/info'))

    def test_break_lock(self):
        ld1 = self.get_lock()
        ld2 = self.get_lock()
        ld1.create()
        ld1.lock_write()
        self.assertRaises(AssertionError, ld1.break_lock)
        orig_factory = breezy.ui.ui_factory
        breezy.ui.ui_factory = breezy.ui.CannedInputUIFactory([True])
        try:
            ld2.break_lock()
            self.assertRaises(LockBroken, ld1.unlock)
        finally:
            breezy.ui.ui_factory = orig_factory

    def test_break_lock_corrupt_info(self):
        """break_lock works even if the info file is corrupt (and tells the UI
        that it is corrupt).
        """
        ld = self.get_lock()
        ld2 = self.get_lock()
        ld.create()
        ld.lock_write()
        ld.transport.put_bytes_non_atomic('test_lock/held/info', b'\x00')

        class LoggingUIFactory(breezy.ui.SilentUIFactory):

            def __init__(self):
                self.prompts = []

            def get_boolean(self, prompt):
                self.prompts.append(('boolean', prompt))
                return True
        ui = LoggingUIFactory()
        self.overrideAttr(breezy.ui, 'ui_factory', ui)
        ld2.break_lock()
        self.assertLength(1, ui.prompts)
        self.assertEqual('boolean', ui.prompts[0][0])
        self.assertStartsWith(ui.prompts[0][1], 'Break (corrupt LockDir')
        self.assertRaises(LockBroken, ld.unlock)

    def test_break_lock_missing_info(self):
        """break_lock works even if the info file is missing (and tells the UI
        that it is corrupt).
        """
        ld = self.get_lock()
        ld2 = self.get_lock()
        ld.create()
        ld.lock_write()
        ld.transport.delete('test_lock/held/info')

        class LoggingUIFactory(breezy.ui.SilentUIFactory):

            def __init__(self):
                self.prompts = []

            def get_boolean(self, prompt):
                self.prompts.append(('boolean', prompt))
                return True
        ui = LoggingUIFactory()
        orig_factory = breezy.ui.ui_factory
        breezy.ui.ui_factory = ui
        try:
            ld2.break_lock()
            self.assertRaises(LockBroken, ld.unlock)
            self.assertLength(0, ui.prompts)
        finally:
            breezy.ui.ui_factory = orig_factory
        del self._lock_actions[:]

    def test_create_missing_base_directory(self):
        """If LockDir.path doesn't exist, it can be created

        Some people manually remove the entire lock/ directory trying
        to unlock a stuck repository/branch/etc. Rather than failing
        after that, just create the lock directory when needed.
        """
        t = self.get_transport()
        lf1 = LockDir(t, 'test_lock')
        lf1.create()
        self.assertTrue(t.has('test_lock'))
        t.rmdir('test_lock')
        self.assertFalse(t.has('test_lock'))
        lf1.lock_write()
        self.assertTrue(t.has('test_lock'))
        self.assertTrue(t.has('test_lock/held/info'))
        lf1.unlock()
        self.assertFalse(t.has('test_lock/held/info'))

    def test_display_form(self):
        ld1 = self.get_lock()
        ld1.create()
        ld1.lock_write()
        try:
            info_list = ld1.peek().to_readable_dict()
        finally:
            ld1.unlock()
        self.assertEqual(info_list['user'], 'jrandom@example.com')
        self.assertIsInstance(info_list['pid'], int)
        self.assertContainsRe(info_list['time_ago'], '^\\d+ seconds? ago$')

    def test_lock_without_email(self):
        global_config = config.GlobalStack()
        global_config.set('email', 'User Identity')
        ld1 = self.get_lock()
        ld1.create()
        ld1.lock_write()
        ld1.unlock()

    def test_lock_permission(self):
        self.requireFeature(features.not_running_as_root)
        if not osutils.supports_posix_readonly():
            raise tests.TestSkipped('Cannot induce a permission failure')
        ld1 = self.get_lock()
        lock_path = ld1.transport.local_abspath('test_lock')
        os.mkdir(lock_path)
        osutils.make_readonly(lock_path)
        self.assertRaises(errors.LockFailed, ld1.attempt_lock)

    def test_lock_by_token(self):
        ld1 = self.get_lock()
        token = ld1.lock_write()
        self.addCleanup(ld1.unlock)
        self.assertNotEqual(None, token)
        ld2 = self.get_lock()
        t2 = ld2.lock_write(token)
        self.addCleanup(ld2.unlock)
        self.assertEqual(token, t2)

    def test_lock_with_buggy_rename(self):
        t = transport.get_transport_from_url('brokenrename+' + self.get_url())
        ld1 = LockDir(t, 'test_lock')
        ld1.create()
        ld1.attempt_lock()
        ld2 = LockDir(t, 'test_lock')
        e = self.assertRaises(errors.LockContention, ld2.attempt_lock)
        ld1.unlock()
        self.assertEqual([], t.list_dir('test_lock'))

    def test_failed_lock_leaves_no_trash(self):
        ld1 = self.get_lock()
        ld2 = self.get_lock()
        ld1.create()
        t = self.get_transport().clone('test_lock')

        def check_dir(a):
            self.assertEqual(a, t.list_dir('.'))
        check_dir([])
        ld1.attempt_lock()
        self.addCleanup(ld1.unlock)
        check_dir(['held'])
        self.assertRaises(errors.LockContention, ld2.attempt_lock)
        check_dir(['held'])

    def test_no_lockdir_info(self):
        """We can cope with empty info files."""
        t = self.get_transport()
        t.mkdir('test_lock')
        t.mkdir('test_lock/held')
        t.put_bytes('test_lock/held/info', b'')
        lf = LockDir(t, 'test_lock')
        info = lf.peek()
        formatted_info = info.to_readable_dict()
        self.assertEqual(dict(user='<unknown>', hostname='<unknown>', pid='<unknown>', time_ago='(unknown)'), formatted_info)

    def test_corrupt_lockdir_info(self):
        """We can cope with corrupt (and thus unparseable) info files."""
        t = self.get_transport()
        t.mkdir('test_lock')
        t.mkdir('test_lock/held')
        t.put_bytes('test_lock/held/info', b'\x00')
        lf = LockDir(t, 'test_lock')
        self.assertRaises(errors.LockCorrupt, lf.peek)
        self.assertRaises((errors.LockCorrupt, errors.LockContention), lf.attempt_lock)
        self.assertRaises(errors.LockCorrupt, lf.validate_token, 'fake token')

    def test_missing_lockdir_info(self):
        """We can cope with absent info files."""
        t = self.get_transport()
        t.mkdir('test_lock')
        t.mkdir('test_lock/held')
        lf = LockDir(t, 'test_lock')
        self.assertEqual(None, lf.peek())
        try:
            lf.attempt_lock()
        except LockContention:
            pass
        else:
            lf.unlock()
        self.assertRaises((errors.TokenMismatch, errors.LockCorrupt), lf.validate_token, 'fake token')