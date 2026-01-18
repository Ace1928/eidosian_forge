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
class TestSensorsAPIs(PsutilTestCase):

    @unittest.skipIf(not HAS_SENSORS_TEMPERATURES, 'not supported')
    def test_sensors_temperatures(self):
        temps = psutil.sensors_temperatures()
        for name, entries in temps.items():
            self.assertIsInstance(name, str)
            for entry in entries:
                self.assertIsInstance(entry.label, str)
                if entry.current is not None:
                    self.assertGreaterEqual(entry.current, 0)
                if entry.high is not None:
                    self.assertGreaterEqual(entry.high, 0)
                if entry.critical is not None:
                    self.assertGreaterEqual(entry.critical, 0)

    @unittest.skipIf(not HAS_SENSORS_TEMPERATURES, 'not supported')
    def test_sensors_temperatures_fahreneit(self):
        d = {'coretemp': [('label', 50.0, 60.0, 70.0)]}
        with mock.patch('psutil._psplatform.sensors_temperatures', return_value=d) as m:
            temps = psutil.sensors_temperatures(fahrenheit=True)['coretemp'][0]
            assert m.called
            self.assertEqual(temps.current, 122.0)
            self.assertEqual(temps.high, 140.0)
            self.assertEqual(temps.critical, 158.0)

    @unittest.skipIf(not HAS_SENSORS_BATTERY, 'not supported')
    @unittest.skipIf(not HAS_BATTERY, 'no battery')
    def test_sensors_battery(self):
        ret = psutil.sensors_battery()
        self.assertGreaterEqual(ret.percent, 0)
        self.assertLessEqual(ret.percent, 100)
        if ret.secsleft not in (psutil.POWER_TIME_UNKNOWN, psutil.POWER_TIME_UNLIMITED):
            self.assertGreaterEqual(ret.secsleft, 0)
        elif ret.secsleft == psutil.POWER_TIME_UNLIMITED:
            self.assertTrue(ret.power_plugged)
        self.assertIsInstance(ret.power_plugged, bool)

    @unittest.skipIf(not HAS_SENSORS_FANS, 'not supported')
    def test_sensors_fans(self):
        fans = psutil.sensors_fans()
        for name, entries in fans.items():
            self.assertIsInstance(name, str)
            for entry in entries:
                self.assertIsInstance(entry.label, str)
                self.assertIsInstance(entry.current, (int, long))
                self.assertGreaterEqual(entry.current, 0)