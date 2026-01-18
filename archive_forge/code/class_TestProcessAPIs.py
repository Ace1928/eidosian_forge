import contextlib
import datetime
import errno
import os
import platform
import pprint
import shutil
import signal
import socket
import sys
import time
import unittest
import psutil
from psutil import AIX
from psutil import BSD
from psutil import FREEBSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._compat import PY3
from psutil._compat import FileNotFoundError
from psutil._compat import long
from psutil.tests import ASCII_FS
from psutil.tests import CI_TESTING
from psutil.tests import DEVNULL
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import GLOBAL_TIMEOUT
from psutil.tests import HAS_BATTERY
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_GETLOADAVG
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import IS_64BIT
from psutil.tests import MACOS_12PLUS
from psutil.tests import PYPY
from psutil.tests import UNICODE_SUFFIX
from psutil.tests import PsutilTestCase
from psutil.tests import check_net_address
from psutil.tests import enum
from psutil.tests import mock
from psutil.tests import retry_on_failure
class TestProcessAPIs(PsutilTestCase):

    def test_process_iter(self):
        self.assertIn(os.getpid(), [x.pid for x in psutil.process_iter()])
        sproc = self.spawn_testproc()
        self.assertIn(sproc.pid, [x.pid for x in psutil.process_iter()])
        p = psutil.Process(sproc.pid)
        p.kill()
        p.wait()
        self.assertNotIn(sproc.pid, [x.pid for x in psutil.process_iter()])
        ls = [x for x in psutil.process_iter()]
        self.assertEqual(sorted(ls, key=lambda x: x.pid), sorted(set(ls), key=lambda x: x.pid))
        with mock.patch('psutil.Process', side_effect=psutil.NoSuchProcess(os.getpid())):
            self.assertEqual(list(psutil.process_iter()), [])
        with mock.patch('psutil.Process', side_effect=psutil.AccessDenied(os.getpid())):
            with self.assertRaises(psutil.AccessDenied):
                list(psutil.process_iter())

    def test_prcess_iter_w_attrs(self):
        for p in psutil.process_iter(attrs=['pid']):
            self.assertEqual(list(p.info.keys()), ['pid'])
        with self.assertRaises(ValueError):
            list(psutil.process_iter(attrs=['foo']))
        with mock.patch('psutil._psplatform.Process.cpu_times', side_effect=psutil.AccessDenied(0, '')) as m:
            for p in psutil.process_iter(attrs=['pid', 'cpu_times']):
                self.assertIsNone(p.info['cpu_times'])
                self.assertGreaterEqual(p.info['pid'], 0)
            assert m.called
        with mock.patch('psutil._psplatform.Process.cpu_times', side_effect=psutil.AccessDenied(0, '')) as m:
            flag = object()
            for p in psutil.process_iter(attrs=['pid', 'cpu_times'], ad_value=flag):
                self.assertIs(p.info['cpu_times'], flag)
                self.assertGreaterEqual(p.info['pid'], 0)
            assert m.called

    @unittest.skipIf(PYPY and WINDOWS, 'spawn_testproc() unreliable on PYPY + WINDOWS')
    def test_wait_procs(self):

        def callback(p):
            pids.append(p.pid)
        pids = []
        sproc1 = self.spawn_testproc()
        sproc2 = self.spawn_testproc()
        sproc3 = self.spawn_testproc()
        procs = [psutil.Process(x.pid) for x in (sproc1, sproc2, sproc3)]
        self.assertRaises(ValueError, psutil.wait_procs, procs, timeout=-1)
        self.assertRaises(TypeError, psutil.wait_procs, procs, callback=1)
        t = time.time()
        gone, alive = psutil.wait_procs(procs, timeout=0.01, callback=callback)
        self.assertLess(time.time() - t, 0.5)
        self.assertEqual(gone, [])
        self.assertEqual(len(alive), 3)
        self.assertEqual(pids, [])
        for p in alive:
            self.assertFalse(hasattr(p, 'returncode'))

        @retry_on_failure(30)
        def test_1(procs, callback):
            gone, alive = psutil.wait_procs(procs, timeout=0.03, callback=callback)
            self.assertEqual(len(gone), 1)
            self.assertEqual(len(alive), 2)
            return (gone, alive)
        sproc3.terminate()
        gone, alive = test_1(procs, callback)
        self.assertIn(sproc3.pid, [x.pid for x in gone])
        if POSIX:
            self.assertEqual(gone.pop().returncode, -signal.SIGTERM)
        else:
            self.assertEqual(gone.pop().returncode, 1)
        self.assertEqual(pids, [sproc3.pid])
        for p in alive:
            self.assertFalse(hasattr(p, 'returncode'))

        @retry_on_failure(30)
        def test_2(procs, callback):
            gone, alive = psutil.wait_procs(procs, timeout=0.03, callback=callback)
            self.assertEqual(len(gone), 3)
            self.assertEqual(len(alive), 0)
            return (gone, alive)
        sproc1.terminate()
        sproc2.terminate()
        gone, alive = test_2(procs, callback)
        self.assertEqual(set(pids), set([sproc1.pid, sproc2.pid, sproc3.pid]))
        for p in gone:
            self.assertTrue(hasattr(p, 'returncode'))

    @unittest.skipIf(PYPY and WINDOWS, 'spawn_testproc() unreliable on PYPY + WINDOWS')
    def test_wait_procs_no_timeout(self):
        sproc1 = self.spawn_testproc()
        sproc2 = self.spawn_testproc()
        sproc3 = self.spawn_testproc()
        procs = [psutil.Process(x.pid) for x in (sproc1, sproc2, sproc3)]
        for p in procs:
            p.terminate()
        psutil.wait_procs(procs)

    def test_pid_exists(self):
        sproc = self.spawn_testproc()
        self.assertTrue(psutil.pid_exists(sproc.pid))
        p = psutil.Process(sproc.pid)
        p.kill()
        p.wait()
        self.assertFalse(psutil.pid_exists(sproc.pid))
        self.assertFalse(psutil.pid_exists(-1))
        self.assertEqual(psutil.pid_exists(0), 0 in psutil.pids())

    def test_pid_exists_2(self):
        pids = psutil.pids()
        for pid in pids:
            try:
                assert psutil.pid_exists(pid)
            except AssertionError:
                time.sleep(0.1)
                self.assertNotIn(pid, psutil.pids())
        pids = range(max(pids) + 15000, max(pids) + 16000)
        for pid in pids:
            self.assertFalse(psutil.pid_exists(pid), msg=pid)