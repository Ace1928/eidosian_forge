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
class TestMiscAPIs(PsutilTestCase):

    def test_boot_time(self):
        bt = psutil.boot_time()
        self.assertIsInstance(bt, float)
        self.assertGreater(bt, 0)
        self.assertLess(bt, time.time())

    @unittest.skipIf(CI_TESTING and (not psutil.users()), 'unreliable on CI')
    def test_users(self):
        users = psutil.users()
        self.assertNotEqual(users, [])
        for user in users:
            with self.subTest(user=user):
                assert user.name
                self.assertIsInstance(user.name, str)
                self.assertIsInstance(user.terminal, (str, type(None)))
                if user.host is not None:
                    self.assertIsInstance(user.host, (str, type(None)))
                user.terminal
                user.host
                self.assertGreater(user.started, 0.0)
                datetime.datetime.fromtimestamp(user.started)
                if WINDOWS or OPENBSD:
                    self.assertIsNone(user.pid)
                else:
                    psutil.Process(user.pid)

    def test_test(self):
        stdout = sys.stdout
        sys.stdout = DEVNULL
        try:
            psutil.test()
        finally:
            sys.stdout = stdout

    def test_os_constants(self):
        names = ['POSIX', 'WINDOWS', 'LINUX', 'MACOS', 'FREEBSD', 'OPENBSD', 'NETBSD', 'BSD', 'SUNOS']
        for name in names:
            self.assertIsInstance(getattr(psutil, name), bool, msg=name)
        if os.name == 'posix':
            assert psutil.POSIX
            assert not psutil.WINDOWS
            names.remove('POSIX')
            if 'linux' in sys.platform.lower():
                assert psutil.LINUX
                names.remove('LINUX')
            elif 'bsd' in sys.platform.lower():
                assert psutil.BSD
                self.assertEqual([psutil.FREEBSD, psutil.OPENBSD, psutil.NETBSD].count(True), 1)
                names.remove('BSD')
                names.remove('FREEBSD')
                names.remove('OPENBSD')
                names.remove('NETBSD')
            elif 'sunos' in sys.platform.lower() or 'solaris' in sys.platform.lower():
                assert psutil.SUNOS
                names.remove('SUNOS')
            elif 'darwin' in sys.platform.lower():
                assert psutil.MACOS
                names.remove('MACOS')
        else:
            assert psutil.WINDOWS
            assert not psutil.POSIX
            names.remove('WINDOWS')
        for name in names:
            self.assertFalse(getattr(psutil, name), msg=name)