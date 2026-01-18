import collections
import errno
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock
from oslotest import base as test_base
from oslo_concurrency.fixture import lockutils as fixtures
from oslo_concurrency import lockutils
from oslo_config import fixture as config
class FileBasedLockingTestCase(test_base.BaseTestCase):

    def setUp(self):
        super(FileBasedLockingTestCase, self).setUp()
        self.lock_dir = tempfile.mkdtemp()

    def test_lock_file_exists(self):
        lock_file = os.path.join(self.lock_dir, 'lock-file')

        @lockutils.synchronized('lock-file', external=True, lock_path=self.lock_dir)
        def foo():
            self.assertTrue(os.path.exists(lock_file))
        foo()

    def test_interprocess_lock(self):
        lock_file = os.path.join(self.lock_dir, 'processlock')
        pid = os.fork()
        if pid:
            start = time.time()
            while not os.path.exists(lock_file):
                if time.time() - start > 5:
                    self.fail('Timed out waiting for child to grab lock')
                time.sleep(0)
            lock1 = lockutils.InterProcessLock('foo')
            lock1.lockfile = open(lock_file, 'w')
            while time.time() - start < 5:
                try:
                    lock1.trylock()
                    lock1.unlock()
                    time.sleep(0)
                except IOError:
                    break
            else:
                self.fail('Never caught expected lock exception')
            os.kill(pid, signal.SIGKILL)
        else:
            try:
                lock2 = lockutils.InterProcessLock('foo')
                lock2.lockfile = open(lock_file, 'w')
                have_lock = False
                while not have_lock:
                    try:
                        lock2.trylock()
                        have_lock = True
                    except IOError:
                        pass
            finally:
                time.sleep(0.5)
                os._exit(0)

    def test_interprocess_nonblocking_external_lock(self):
        """Check that we're not actually blocking between processes."""
        nb_calls = multiprocessing.Value('i', 0)

        @lockutils.synchronized('foo', blocking=False, external=True, lock_path=self.lock_dir)
        def foo(param):
            """Simulate a long-running operation in a process."""
            param.value += 1
            time.sleep(0.5)

        def other(param):
            foo(param)
        process = multiprocessing.Process(target=other, args=(nb_calls,))
        process.start()
        start = time.time()
        while not os.path.exists(os.path.join(self.lock_dir, 'foo')):
            if time.time() - start > 5:
                self.fail('Timed out waiting for process to grab lock')
            time.sleep(0)
        process1 = multiprocessing.Process(target=other, args=(nb_calls,))
        process1.start()
        process1.join()
        process.join()
        self.assertEqual(1, nb_calls.value)

    def test_interthread_external_lock(self):
        call_list = []

        @lockutils.synchronized('foo', external=True, lock_path=self.lock_dir)
        def foo(param):
            """Simulate a long-running threaded operation."""
            call_list.append(param)
            time.sleep(0.5)
            call_list.append(param)

        def other(param):
            foo(param)
        thread = threading.Thread(target=other, args=('other',))
        thread.start()
        start = time.time()
        while not os.path.exists(os.path.join(self.lock_dir, 'foo')):
            if time.time() - start > 5:
                self.fail('Timed out waiting for thread to grab lock')
            time.sleep(0)
        thread1 = threading.Thread(target=other, args=('main',))
        thread1.start()
        thread1.join()
        thread.join()
        self.assertEqual(['other', 'other', 'main', 'main'], call_list)

    def test_interthread_nonblocking_external_lock(self):
        call_list = []

        @lockutils.synchronized('foo', external=True, blocking=False, lock_path=self.lock_dir)
        def foo(param):
            """Simulate a long-running threaded operation."""
            call_list.append(param)
            time.sleep(0.5)
            call_list.append(param)

        def other(param):
            foo(param)
        thread = threading.Thread(target=other, args=('other',))
        thread.start()
        start = time.time()
        while not os.path.exists(os.path.join(self.lock_dir, 'foo')):
            if time.time() - start > 5:
                self.fail('Timed out waiting for thread to grab lock')
            time.sleep(0)
        thread1 = threading.Thread(target=other, args=('main',))
        thread1.start()
        thread1.join()
        thread.join()
        self.assertEqual(['other', 'other'], call_list)

    def test_interthread_nonblocking_internal_lock(self):
        call_list = []

        @lockutils.synchronized('foo', blocking=False, lock_path=self.lock_dir)
        def foo(param):
            call_list.append(param)
            time.sleep(0.5)
            call_list.append(param)

        def other(param):
            foo(param)
        thread = threading.Thread(target=other, args=('other',))
        thread.start()
        start = time.time()
        while not call_list:
            if time.time() - start > 5:
                self.fail('Timed out waiting for thread to grab lock')
            time.sleep(0)
        thread1 = threading.Thread(target=other, args=('main',))
        thread1.start()
        thread1.join()
        thread.join()
        self.assertEqual(['other', 'other'], call_list)

    def test_non_destructive(self):
        lock_file = os.path.join(self.lock_dir, 'not-destroyed')
        with open(lock_file, 'w') as f:
            f.write('test')
        with lockutils.lock('not-destroyed', external=True, lock_path=self.lock_dir):
            with open(lock_file) as f:
                self.assertEqual('test', f.read())