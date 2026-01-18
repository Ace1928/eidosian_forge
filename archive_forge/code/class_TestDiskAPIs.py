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
class TestDiskAPIs(PsutilTestCase):

    @unittest.skipIf(PYPY and (not IS_64BIT), 'unreliable on PYPY32 + 32BIT')
    def test_disk_usage(self):
        usage = psutil.disk_usage(os.getcwd())
        self.assertEqual(usage._fields, ('total', 'used', 'free', 'percent'))
        assert usage.total > 0, usage
        assert usage.used > 0, usage
        assert usage.free > 0, usage
        assert usage.total > usage.used, usage
        assert usage.total > usage.free, usage
        assert 0 <= usage.percent <= 100, usage.percent
        if hasattr(shutil, 'disk_usage'):
            shutil_usage = shutil.disk_usage(os.getcwd())
            tolerance = 5 * 1024 * 1024
            self.assertEqual(usage.total, shutil_usage.total)
            self.assertAlmostEqual(usage.free, shutil_usage.free, delta=tolerance)
            if not MACOS_12PLUS:
                self.assertAlmostEqual(usage.used, shutil_usage.used, delta=tolerance)
        fname = self.get_testfn()
        with self.assertRaises(FileNotFoundError):
            psutil.disk_usage(fname)

    @unittest.skipIf(not ASCII_FS, 'not an ASCII fs')
    def test_disk_usage_unicode(self):
        with self.assertRaises(UnicodeEncodeError):
            psutil.disk_usage(UNICODE_SUFFIX)

    def test_disk_usage_bytes(self):
        psutil.disk_usage(b'.')

    def test_disk_partitions(self):

        def check_ntuple(nt):
            self.assertIsInstance(nt.device, str)
            self.assertIsInstance(nt.mountpoint, str)
            self.assertIsInstance(nt.fstype, str)
            self.assertIsInstance(nt.opts, str)
            self.assertIsInstance(nt.maxfile, (int, type(None)))
            self.assertIsInstance(nt.maxpath, (int, type(None)))
            if nt.maxfile is not None and (not GITHUB_ACTIONS):
                self.assertGreater(nt.maxfile, 0)
            if nt.maxpath is not None:
                self.assertGreater(nt.maxpath, 0)
        ls = psutil.disk_partitions(all=False)
        self.assertTrue(ls, msg=ls)
        for disk in ls:
            check_ntuple(disk)
            if WINDOWS and 'cdrom' in disk.opts:
                continue
            if not POSIX:
                assert os.path.exists(disk.device), disk
            else:
                disk.device
            assert os.path.exists(disk.mountpoint), disk
            assert disk.fstype, disk
        ls = psutil.disk_partitions(all=True)
        self.assertTrue(ls, msg=ls)
        for disk in psutil.disk_partitions(all=True):
            check_ntuple(disk)
            if not WINDOWS and disk.mountpoint:
                try:
                    os.stat(disk.mountpoint)
                except OSError as err:
                    if GITHUB_ACTIONS and MACOS and (err.errno == errno.EIO):
                        continue
                    if err.errno not in (errno.EPERM, errno.EACCES):
                        raise
                else:
                    assert os.path.exists(disk.mountpoint), disk

        def find_mount_point(path):
            path = os.path.abspath(path)
            while not os.path.ismount(path):
                path = os.path.dirname(path)
            return path.lower()
        mount = find_mount_point(__file__)
        mounts = [x.mountpoint.lower() for x in psutil.disk_partitions(all=True) if x.mountpoint]
        self.assertIn(mount, mounts)

    @unittest.skipIf(LINUX and (not os.path.exists('/proc/diskstats')), '/proc/diskstats not available on this linux version')
    @unittest.skipIf(CI_TESTING and (not psutil.disk_io_counters()), 'unreliable on CI')
    def test_disk_io_counters(self):

        def check_ntuple(nt):
            self.assertEqual(nt[0], nt.read_count)
            self.assertEqual(nt[1], nt.write_count)
            self.assertEqual(nt[2], nt.read_bytes)
            self.assertEqual(nt[3], nt.write_bytes)
            if not (OPENBSD or NETBSD):
                self.assertEqual(nt[4], nt.read_time)
                self.assertEqual(nt[5], nt.write_time)
                if LINUX:
                    self.assertEqual(nt[6], nt.read_merged_count)
                    self.assertEqual(nt[7], nt.write_merged_count)
                    self.assertEqual(nt[8], nt.busy_time)
                elif FREEBSD:
                    self.assertEqual(nt[6], nt.busy_time)
            for name in nt._fields:
                assert getattr(nt, name) >= 0, nt
        ret = psutil.disk_io_counters(perdisk=False)
        assert ret is not None, 'no disks on this system?'
        check_ntuple(ret)
        ret = psutil.disk_io_counters(perdisk=True)
        self.assertEqual(len(ret), len(set(ret)))
        for key in ret:
            assert key, key
            check_ntuple(ret[key])

    def test_disk_io_counters_no_disks(self):
        with mock.patch('psutil._psplatform.disk_io_counters', return_value={}) as m:
            self.assertIsNone(psutil.disk_io_counters(perdisk=False))
            self.assertEqual(psutil.disk_io_counters(perdisk=True), {})
            assert m.called