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
class TestProcessAgainstStatus(PsutilTestCase):
    """/proc/pid/stat and /proc/pid/status have many values in common.
    Whenever possible, psutil uses /proc/pid/stat (it's faster).
    For all those cases we check that the value found in
    /proc/pid/stat (by psutil) matches the one found in
    /proc/pid/status.
    """

    @classmethod
    def setUpClass(cls):
        cls.proc = psutil.Process()

    def read_status_file(self, linestart):
        with psutil._psplatform.open_text('/proc/%s/status' % self.proc.pid) as f:
            for line in f:
                line = line.strip()
                if line.startswith(linestart):
                    value = line.partition('\t')[2]
                    try:
                        return int(value)
                    except ValueError:
                        return value
            raise ValueError("can't find %r" % linestart)

    def test_name(self):
        value = self.read_status_file('Name:')
        self.assertEqual(self.proc.name(), value)

    def test_status(self):
        value = self.read_status_file('State:')
        value = value[value.find('(') + 1:value.rfind(')')]
        value = value.replace(' ', '-')
        self.assertEqual(self.proc.status(), value)

    def test_ppid(self):
        value = self.read_status_file('PPid:')
        self.assertEqual(self.proc.ppid(), value)

    def test_num_threads(self):
        value = self.read_status_file('Threads:')
        self.assertEqual(self.proc.num_threads(), value)

    def test_uids(self):
        value = self.read_status_file('Uid:')
        value = tuple(map(int, value.split()[1:4]))
        self.assertEqual(self.proc.uids(), value)

    def test_gids(self):
        value = self.read_status_file('Gid:')
        value = tuple(map(int, value.split()[1:4]))
        self.assertEqual(self.proc.gids(), value)

    @retry_on_failure()
    def test_num_ctx_switches(self):
        value = self.read_status_file('voluntary_ctxt_switches:')
        self.assertEqual(self.proc.num_ctx_switches().voluntary, value)
        value = self.read_status_file('nonvoluntary_ctxt_switches:')
        self.assertEqual(self.proc.num_ctx_switches().involuntary, value)

    def test_cpu_affinity(self):
        value = self.read_status_file('Cpus_allowed_list:')
        if '-' in str(value):
            min_, max_ = map(int, value.split('-'))
            self.assertEqual(self.proc.cpu_affinity(), list(range(min_, max_ + 1)))

    def test_cpu_affinity_eligible_cpus(self):
        value = self.read_status_file('Cpus_allowed_list:')
        with mock.patch('psutil._pslinux.per_cpu_times') as m:
            self.proc._proc._get_eligible_cpus()
        if '-' in str(value):
            assert not m.called
        else:
            assert m.called