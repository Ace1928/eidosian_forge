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
class TestSystemNetIfAddrs(PsutilTestCase):

    def test_ips(self):
        for name, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == psutil.AF_LINK:
                    self.assertEqual(addr.address, get_mac_address(name))
                elif addr.family == socket.AF_INET:
                    self.assertEqual(addr.address, get_ipv4_address(name))
                    self.assertEqual(addr.netmask, get_ipv4_netmask(name))
                    if addr.broadcast is not None:
                        self.assertEqual(addr.broadcast, get_ipv4_broadcast(name))
                    else:
                        self.assertEqual(get_ipv4_broadcast(name), '0.0.0.0')
                elif addr.family == socket.AF_INET6:
                    address = addr.address.split('%')[0]
                    self.assertIn(address, get_ipv6_addresses(name))