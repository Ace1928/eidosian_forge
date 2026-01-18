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
class TestSystemNetIfStats(PsutilTestCase):

    @unittest.skipIf(not which('ifconfig'), 'ifconfig utility not available')
    def test_against_ifconfig(self):
        for name, stats in psutil.net_if_stats().items():
            try:
                out = sh('ifconfig %s' % name)
            except RuntimeError:
                pass
            else:
                self.assertEqual(stats.isup, 'RUNNING' in out, msg=out)
                self.assertEqual(stats.mtu, int(re.findall('(?i)MTU[: ](\\d+)', out)[0]))

    def test_mtu(self):
        for name, stats in psutil.net_if_stats().items():
            with open('/sys/class/net/%s/mtu' % name) as f:
                self.assertEqual(stats.mtu, int(f.read().strip()))

    @unittest.skipIf(not which('ifconfig'), 'ifconfig utility not available')
    def test_flags(self):
        matches_found = 0
        for name, stats in psutil.net_if_stats().items():
            try:
                out = sh('ifconfig %s' % name)
            except RuntimeError:
                pass
            else:
                match = re.search('flags=(\\d+)?<(.*?)>', out)
                if match and len(match.groups()) >= 2:
                    matches_found += 1
                    ifconfig_flags = set(match.group(2).lower().split(','))
                    psutil_flags = set(stats.flags.split(','))
                    self.assertEqual(ifconfig_flags, psutil_flags)
                else:
                    match = re.search('(.*)  MTU:(\\d+)  Metric:(\\d+)', out)
                    if match and len(match.groups()) >= 3:
                        matches_found += 1
                        ifconfig_flags = set(match.group(1).lower().split())
                        psutil_flags = set(stats.flags.split(','))
                        self.assertEqual(ifconfig_flags, psutil_flags)
        if not matches_found:
            raise self.fail('no matches were found')