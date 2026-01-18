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
class TestLoadAvg(PsutilTestCase):

    @unittest.skipIf(not HAS_GETLOADAVG, 'not supported')
    def test_getloadavg(self):
        psutil_value = psutil.getloadavg()
        with open('/proc/loadavg') as f:
            proc_value = f.read().split()
        self.assertAlmostEqual(float(proc_value[0]), psutil_value[0], delta=1)
        self.assertAlmostEqual(float(proc_value[1]), psutil_value[1], delta=1)
        self.assertAlmostEqual(float(proc_value[2]), psutil_value[2], delta=1)