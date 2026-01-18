import platform
import signal
import unittest
import psutil
from psutil import AIX
from psutil import FREEBSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._compat import long
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import PYPY
from psutil.tests import SKIP_SYSCONS
from psutil.tests import PsutilTestCase
from psutil.tests import create_sockets
from psutil.tests import enum
from psutil.tests import is_namedtuple
from psutil.tests import kernel_version
class TestAvailSystemAPIs(PsutilTestCase):

    def test_win_service_iter(self):
        self.assertEqual(hasattr(psutil, 'win_service_iter'), WINDOWS)

    def test_win_service_get(self):
        self.assertEqual(hasattr(psutil, 'win_service_get'), WINDOWS)

    def test_cpu_freq(self):
        self.assertEqual(hasattr(psutil, 'cpu_freq'), LINUX or MACOS or WINDOWS or FREEBSD or OPENBSD)

    def test_sensors_temperatures(self):
        self.assertEqual(hasattr(psutil, 'sensors_temperatures'), LINUX or FREEBSD)

    def test_sensors_fans(self):
        self.assertEqual(hasattr(psutil, 'sensors_fans'), LINUX)

    def test_battery(self):
        self.assertEqual(hasattr(psutil, 'sensors_battery'), LINUX or WINDOWS or FREEBSD or MACOS)