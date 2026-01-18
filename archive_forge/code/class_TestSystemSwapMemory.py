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
class TestSystemSwapMemory(PsutilTestCase):

    @staticmethod
    def meminfo_has_swap_info():
        """Return True if /proc/meminfo provides swap metrics."""
        with open('/proc/meminfo') as f:
            data = f.read()
        return 'SwapTotal:' in data and 'SwapFree:' in data

    def test_total(self):
        free_value = free_swap().total
        psutil_value = psutil.swap_memory().total
        return self.assertAlmostEqual(free_value, psutil_value, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_used(self):
        free_value = free_swap().used
        psutil_value = psutil.swap_memory().used
        return self.assertAlmostEqual(free_value, psutil_value, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_free(self):
        free_value = free_swap().free
        psutil_value = psutil.swap_memory().free
        return self.assertAlmostEqual(free_value, psutil_value, delta=TOLERANCE_SYS_MEM)

    def test_missing_sin_sout(self):
        with mock.patch('psutil._common.open', create=True) as m:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                ret = psutil.swap_memory()
                assert m.called
                self.assertEqual(len(ws), 1)
                w = ws[0]
                self.assertIn("'sin' and 'sout' swap memory stats couldn't be determined", str(w.message))
                self.assertEqual(ret.sin, 0)
                self.assertEqual(ret.sout, 0)

    def test_no_vmstat_mocked(self):
        with mock_open_exception('/proc/vmstat', IOError(errno.ENOENT, 'no such file or directory')) as m:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                ret = psutil.swap_memory()
                assert m.called
                self.assertEqual(len(ws), 1)
                w = ws[0]
                self.assertIn("'sin' and 'sout' swap memory stats couldn't be determined and were set to 0", str(w.message))
                self.assertEqual(ret.sin, 0)
                self.assertEqual(ret.sout, 0)

    def test_meminfo_against_sysinfo(self):
        if not self.meminfo_has_swap_info():
            return unittest.skip('/proc/meminfo has no swap metrics')
        with mock.patch('psutil._pslinux.cext.linux_sysinfo') as m:
            swap = psutil.swap_memory()
        assert not m.called
        import psutil._psutil_linux as cext
        _, _, _, _, total, free, unit_multiplier = cext.linux_sysinfo()
        total *= unit_multiplier
        free *= unit_multiplier
        self.assertEqual(swap.total, total)
        self.assertAlmostEqual(swap.free, free, delta=TOLERANCE_SYS_MEM)

    def test_emulate_meminfo_has_no_metrics(self):
        with mock_open_content({'/proc/meminfo': b''}) as m:
            psutil.swap_memory()
            assert m.called