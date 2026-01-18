import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
class TestLockDirHooks(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self._calls = []

    def get_lock(self):
        return LockDir(self.get_transport(), 'test_lock')

    def record_hook(self, result):
        self._calls.append(result)

    def test_LockDir_acquired_success(self):
        LockDir.hooks.install_named_hook('lock_acquired', self.record_hook, 'record_hook')
        ld = self.get_lock()
        ld.create()
        self.assertEqual([], self._calls)
        result = ld.attempt_lock()
        lock_path = ld.transport.abspath(ld.path)
        self.assertEqual([lock.LockResult(lock_path, result)], self._calls)
        ld.unlock()
        self.assertEqual([lock.LockResult(lock_path, result)], self._calls)

    def test_LockDir_acquired_fail(self):
        ld = self.get_lock()
        ld.create()
        ld2 = self.get_lock()
        ld2.attempt_lock()
        LockDir.hooks.install_named_hook('lock_acquired', self.record_hook, 'record_hook')
        self.assertRaises(errors.LockContention, ld.attempt_lock)
        self.assertEqual([], self._calls)
        ld2.unlock()
        self.assertEqual([], self._calls)

    def test_LockDir_released_success(self):
        LockDir.hooks.install_named_hook('lock_released', self.record_hook, 'record_hook')
        ld = self.get_lock()
        ld.create()
        self.assertEqual([], self._calls)
        result = ld.attempt_lock()
        self.assertEqual([], self._calls)
        ld.unlock()
        lock_path = ld.transport.abspath(ld.path)
        self.assertEqual([lock.LockResult(lock_path, result)], self._calls)

    def test_LockDir_released_fail(self):
        ld = self.get_lock()
        ld.create()
        ld2 = self.get_lock()
        ld.attempt_lock()
        ld2.force_break(ld2.peek())
        LockDir.hooks.install_named_hook('lock_released', self.record_hook, 'record_hook')
        self.assertRaises(LockBroken, ld.unlock)
        self.assertEqual([], self._calls)

    def test_LockDir_broken_success(self):
        ld = self.get_lock()
        ld.create()
        ld2 = self.get_lock()
        result = ld.attempt_lock()
        LockDir.hooks.install_named_hook('lock_broken', self.record_hook, 'record_hook')
        ld2.force_break(ld2.peek())
        lock_path = ld.transport.abspath(ld.path)
        self.assertEqual([lock.LockResult(lock_path, result)], self._calls)

    def test_LockDir_broken_failure(self):
        ld = self.get_lock()
        ld.create()
        ld2 = self.get_lock()
        result = ld.attempt_lock()
        holder_info = ld2.peek()
        ld.unlock()
        LockDir.hooks.install_named_hook('lock_broken', self.record_hook, 'record_hook')
        ld2.force_break(holder_info)
        lock_path = ld.transport.abspath(ld.path)
        self.assertEqual([], self._calls)