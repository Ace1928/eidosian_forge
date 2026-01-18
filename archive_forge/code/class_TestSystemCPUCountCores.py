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
class TestSystemCPUCountCores(PsutilTestCase):

    @unittest.skipIf(not which('lscpu'), 'lscpu utility not available')
    def test_against_lscpu(self):
        out = sh('lscpu -p')
        core_ids = set()
        for line in out.split('\n'):
            if not line.startswith('#'):
                fields = line.split(',')
                core_ids.add(fields[1])
        self.assertEqual(psutil.cpu_count(logical=False), len(core_ids))

    def test_method_2(self):
        meth_1 = psutil._pslinux.cpu_count_cores()
        with mock.patch('glob.glob', return_value=[]) as m:
            meth_2 = psutil._pslinux.cpu_count_cores()
            assert m.called
        if meth_1 is not None:
            self.assertEqual(meth_1, meth_2)

    def test_emulate_none(self):
        with mock.patch('glob.glob', return_value=[]) as m1:
            with mock.patch('psutil._common.open', create=True) as m2:
                self.assertIsNone(psutil._pslinux.cpu_count_cores())
        assert m1.called
        assert m2.called