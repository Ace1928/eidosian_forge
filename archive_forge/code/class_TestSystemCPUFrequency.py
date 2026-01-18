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
class TestSystemCPUFrequency(PsutilTestCase):

    @unittest.skipIf(not HAS_CPU_FREQ, 'not supported')
    def test_emulate_use_second_file(self):

        def path_exists_mock(path):
            if path.startswith('/sys/devices/system/cpu/cpufreq/policy'):
                return False
            else:
                return orig_exists(path)
        orig_exists = os.path.exists
        with mock.patch('os.path.exists', side_effect=path_exists_mock, create=True):
            assert psutil.cpu_freq()

    @unittest.skipIf(not HAS_CPU_FREQ, 'not supported')
    def test_emulate_use_cpuinfo(self):

        def path_exists_mock(path):
            if path.startswith('/sys/devices/system/cpu/'):
                return False
            else:
                return os_path_exists(path)
        os_path_exists = os.path.exists
        try:
            with mock.patch('os.path.exists', side_effect=path_exists_mock):
                reload_module(psutil._pslinux)
                ret = psutil.cpu_freq()
                assert ret, ret
                self.assertEqual(ret.max, 0.0)
                self.assertEqual(ret.min, 0.0)
                for freq in psutil.cpu_freq(percpu=True):
                    self.assertEqual(freq.max, 0.0)
                    self.assertEqual(freq.min, 0.0)
        finally:
            reload_module(psutil._pslinux)
            reload_module(psutil)

    @unittest.skipIf(not HAS_CPU_FREQ, 'not supported')
    def test_emulate_data(self):

        def open_mock(name, *args, **kwargs):
            if name.endswith('/scaling_cur_freq') and name.startswith('/sys/devices/system/cpu/cpufreq/policy'):
                return io.BytesIO(b'500000')
            elif name.endswith('/scaling_min_freq') and name.startswith('/sys/devices/system/cpu/cpufreq/policy'):
                return io.BytesIO(b'600000')
            elif name.endswith('/scaling_max_freq') and name.startswith('/sys/devices/system/cpu/cpufreq/policy'):
                return io.BytesIO(b'700000')
            elif name == '/proc/cpuinfo':
                return io.BytesIO(b'cpu MHz     : 500')
            else:
                return orig_open(name, *args, **kwargs)
        orig_open = open
        patch_point = 'builtins.open' if PY3 else '__builtin__.open'
        with mock.patch(patch_point, side_effect=open_mock):
            with mock.patch('os.path.exists', return_value=True):
                freq = psutil.cpu_freq()
                self.assertEqual(freq.current, 500.0)
                if freq.min != 0.0:
                    self.assertEqual(freq.min, 600.0)
                if freq.max != 0.0:
                    self.assertEqual(freq.max, 700.0)

    @unittest.skipIf(not HAS_CPU_FREQ, 'not supported')
    def test_emulate_multi_cpu(self):

        def open_mock(name, *args, **kwargs):
            n = name
            if n.endswith('/scaling_cur_freq') and n.startswith('/sys/devices/system/cpu/cpufreq/policy0'):
                return io.BytesIO(b'100000')
            elif n.endswith('/scaling_min_freq') and n.startswith('/sys/devices/system/cpu/cpufreq/policy0'):
                return io.BytesIO(b'200000')
            elif n.endswith('/scaling_max_freq') and n.startswith('/sys/devices/system/cpu/cpufreq/policy0'):
                return io.BytesIO(b'300000')
            elif n.endswith('/scaling_cur_freq') and n.startswith('/sys/devices/system/cpu/cpufreq/policy1'):
                return io.BytesIO(b'400000')
            elif n.endswith('/scaling_min_freq') and n.startswith('/sys/devices/system/cpu/cpufreq/policy1'):
                return io.BytesIO(b'500000')
            elif n.endswith('/scaling_max_freq') and n.startswith('/sys/devices/system/cpu/cpufreq/policy1'):
                return io.BytesIO(b'600000')
            elif name == '/proc/cpuinfo':
                return io.BytesIO(b'cpu MHz     : 100\ncpu MHz     : 400')
            else:
                return orig_open(name, *args, **kwargs)
        orig_open = open
        patch_point = 'builtins.open' if PY3 else '__builtin__.open'
        with mock.patch(patch_point, side_effect=open_mock):
            with mock.patch('os.path.exists', return_value=True):
                with mock.patch('psutil._pslinux.cpu_count_logical', return_value=2):
                    freq = psutil.cpu_freq(percpu=True)
                    self.assertEqual(freq[0].current, 100.0)
                    if freq[0].min != 0.0:
                        self.assertEqual(freq[0].min, 200.0)
                    if freq[0].max != 0.0:
                        self.assertEqual(freq[0].max, 300.0)
                    self.assertEqual(freq[1].current, 400.0)
                    if freq[1].min != 0.0:
                        self.assertEqual(freq[1].min, 500.0)
                    if freq[1].max != 0.0:
                        self.assertEqual(freq[1].max, 600.0)

    @unittest.skipIf(not HAS_CPU_FREQ, 'not supported')
    def test_emulate_no_scaling_cur_freq_file(self):

        def open_mock(name, *args, **kwargs):
            if name.endswith('/scaling_cur_freq'):
                raise IOError(errno.ENOENT, '')
            elif name.endswith('/cpuinfo_cur_freq'):
                return io.BytesIO(b'200000')
            elif name == '/proc/cpuinfo':
                return io.BytesIO(b'cpu MHz     : 200')
            else:
                return orig_open(name, *args, **kwargs)
        orig_open = open
        patch_point = 'builtins.open' if PY3 else '__builtin__.open'
        with mock.patch(patch_point, side_effect=open_mock):
            with mock.patch('os.path.exists', return_value=True):
                with mock.patch('psutil._pslinux.cpu_count_logical', return_value=1):
                    freq = psutil.cpu_freq()
                    self.assertEqual(freq.current, 200)