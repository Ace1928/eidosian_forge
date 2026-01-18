from __future__ import division
import collections
import contextlib
import errno
import glob
import io
import os
import re
import shutil
import socket
import struct
import textwrap
import time
import unittest
import warnings
import psutil
from psutil import LINUX
from psutil._compat import PY3
from psutil._compat import FileNotFoundError
from psutil._compat import basestring
from psutil._compat import u
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import GLOBAL_TIMEOUT
from psutil.tests import HAS_BATTERY
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_GETLOADAVG
from psutil.tests import HAS_RLIMIT
from psutil.tests import PYPY
from psutil.tests import TOLERANCE_DISK_USAGE
from psutil.tests import TOLERANCE_SYS_MEM
from psutil.tests import PsutilTestCase
from psutil.tests import ThreadTask
from psutil.tests import call_until
from psutil.tests import mock
from psutil.tests import reload_module
from psutil.tests import retry_on_failure
from psutil.tests import safe_rmpath
from psutil.tests import sh
from psutil.tests import skip_on_not_implemented
from psutil.tests import which
@unittest.skipIf(not LINUX, 'LINUX only')
class TestSystemDiskIoCounters(PsutilTestCase):

    def test_emulate_kernel_2_4(self):
        content = '   3     0   1 hda 2 3 4 5 6 7 8 9 10 11 12'
        with mock_open_content({'/proc/diskstats': content}):
            with mock.patch('psutil._pslinux.is_storage_device', return_value=True):
                ret = psutil.disk_io_counters(nowrap=False)
                self.assertEqual(ret.read_count, 1)
                self.assertEqual(ret.read_merged_count, 2)
                self.assertEqual(ret.read_bytes, 3 * SECTOR_SIZE)
                self.assertEqual(ret.read_time, 4)
                self.assertEqual(ret.write_count, 5)
                self.assertEqual(ret.write_merged_count, 6)
                self.assertEqual(ret.write_bytes, 7 * SECTOR_SIZE)
                self.assertEqual(ret.write_time, 8)
                self.assertEqual(ret.busy_time, 10)

    def test_emulate_kernel_2_6_full(self):
        content = '   3    0   hda 1 2 3 4 5 6 7 8 9 10 11'
        with mock_open_content({'/proc/diskstats': content}):
            with mock.patch('psutil._pslinux.is_storage_device', return_value=True):
                ret = psutil.disk_io_counters(nowrap=False)
                self.assertEqual(ret.read_count, 1)
                self.assertEqual(ret.read_merged_count, 2)
                self.assertEqual(ret.read_bytes, 3 * SECTOR_SIZE)
                self.assertEqual(ret.read_time, 4)
                self.assertEqual(ret.write_count, 5)
                self.assertEqual(ret.write_merged_count, 6)
                self.assertEqual(ret.write_bytes, 7 * SECTOR_SIZE)
                self.assertEqual(ret.write_time, 8)
                self.assertEqual(ret.busy_time, 10)

    def test_emulate_kernel_2_6_limited(self):
        with mock_open_content({'/proc/diskstats': '   3    1   hda 1 2 3 4'}):
            with mock.patch('psutil._pslinux.is_storage_device', return_value=True):
                ret = psutil.disk_io_counters(nowrap=False)
                self.assertEqual(ret.read_count, 1)
                self.assertEqual(ret.read_bytes, 2 * SECTOR_SIZE)
                self.assertEqual(ret.write_count, 3)
                self.assertEqual(ret.write_bytes, 4 * SECTOR_SIZE)
                self.assertEqual(ret.read_merged_count, 0)
                self.assertEqual(ret.read_time, 0)
                self.assertEqual(ret.write_merged_count, 0)
                self.assertEqual(ret.write_time, 0)
                self.assertEqual(ret.busy_time, 0)

    def test_emulate_include_partitions(self):
        content = textwrap.dedent('            3    0   nvme0n1 1 2 3 4 5 6 7 8 9 10 11\n            3    0   nvme0n1p1 1 2 3 4 5 6 7 8 9 10 11\n            ')
        with mock_open_content({'/proc/diskstats': content}):
            with mock.patch('psutil._pslinux.is_storage_device', return_value=False):
                ret = psutil.disk_io_counters(perdisk=True, nowrap=False)
                self.assertEqual(len(ret), 2)
                self.assertEqual(ret['nvme0n1'].read_count, 1)
                self.assertEqual(ret['nvme0n1p1'].read_count, 1)
                self.assertEqual(ret['nvme0n1'].write_count, 5)
                self.assertEqual(ret['nvme0n1p1'].write_count, 5)

    def test_emulate_exclude_partitions(self):
        content = textwrap.dedent('            3    0   nvme0n1 1 2 3 4 5 6 7 8 9 10 11\n            3    0   nvme0n1p1 1 2 3 4 5 6 7 8 9 10 11\n            ')
        with mock_open_content({'/proc/diskstats': content}):
            with mock.patch('psutil._pslinux.is_storage_device', return_value=False):
                ret = psutil.disk_io_counters(perdisk=False, nowrap=False)
                self.assertIsNone(ret)

        def is_storage_device(name):
            return name == 'nvme0n1'
        content = textwrap.dedent('            3    0   nvme0n1 1 2 3 4 5 6 7 8 9 10 11\n            3    0   nvme0n1p1 1 2 3 4 5 6 7 8 9 10 11\n            ')
        with mock_open_content({'/proc/diskstats': content}):
            with mock.patch('psutil._pslinux.is_storage_device', create=True, side_effect=is_storage_device):
                ret = psutil.disk_io_counters(perdisk=False, nowrap=False)
                self.assertEqual(ret.read_count, 1)
                self.assertEqual(ret.write_count, 5)

    def test_emulate_use_sysfs(self):

        def exists(path):
            if path == '/proc/diskstats':
                return False
            return True
        wprocfs = psutil.disk_io_counters(perdisk=True)
        with mock.patch('psutil._pslinux.os.path.exists', create=True, side_effect=exists):
            wsysfs = psutil.disk_io_counters(perdisk=True)
        self.assertEqual(len(wprocfs), len(wsysfs))

    def test_emulate_not_impl(self):

        def exists(path):
            return False
        with mock.patch('psutil._pslinux.os.path.exists', create=True, side_effect=exists):
            self.assertRaises(NotImplementedError, psutil.disk_io_counters)