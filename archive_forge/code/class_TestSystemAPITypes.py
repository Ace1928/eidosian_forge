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
class TestSystemAPITypes(PsutilTestCase):
    """Check the return types of system related APIs.
    Mainly we want to test we never return unicode on Python 2, see:
    https://github.com/giampaolo/psutil/issues/1039.
    """

    @classmethod
    def setUpClass(cls):
        cls.proc = psutil.Process()

    def assert_ntuple_of_nums(self, nt, type_=float, gezero=True):
        assert is_namedtuple(nt)
        for n in nt:
            self.assertIsInstance(n, type_)
            if gezero:
                self.assertGreaterEqual(n, 0)

    def test_cpu_times(self):
        self.assert_ntuple_of_nums(psutil.cpu_times())
        for nt in psutil.cpu_times(percpu=True):
            self.assert_ntuple_of_nums(nt)

    def test_cpu_percent(self):
        self.assertIsInstance(psutil.cpu_percent(interval=None), float)
        self.assertIsInstance(psutil.cpu_percent(interval=1e-05), float)

    def test_cpu_times_percent(self):
        self.assert_ntuple_of_nums(psutil.cpu_times_percent(interval=None))
        self.assert_ntuple_of_nums(psutil.cpu_times_percent(interval=0.0001))

    def test_cpu_count(self):
        self.assertIsInstance(psutil.cpu_count(), int)

    @unittest.skipIf(MACOS and platform.machine() == 'arm64', 'skipped due to #1892')
    @unittest.skipIf(not HAS_CPU_FREQ, 'not supported')
    def test_cpu_freq(self):
        if psutil.cpu_freq() is None:
            raise self.skipTest('cpu_freq() returns None')
        self.assert_ntuple_of_nums(psutil.cpu_freq(), type_=(float, int, long))

    def test_disk_io_counters(self):
        for k, v in psutil.disk_io_counters(perdisk=True).items():
            self.assertIsInstance(k, str)
            self.assert_ntuple_of_nums(v, type_=(int, long))

    def test_disk_partitions(self):
        for disk in psutil.disk_partitions():
            self.assertIsInstance(disk.device, str)
            self.assertIsInstance(disk.mountpoint, str)
            self.assertIsInstance(disk.fstype, str)
            self.assertIsInstance(disk.opts, str)
            self.assertIsInstance(disk.maxfile, (int, type(None)))
            self.assertIsInstance(disk.maxpath, (int, type(None)))

    @unittest.skipIf(SKIP_SYSCONS, 'requires root')
    def test_net_connections(self):
        with create_sockets():
            ret = psutil.net_connections('all')
            self.assertEqual(len(ret), len(set(ret)))
            for conn in ret:
                assert is_namedtuple(conn)

    def test_net_if_addrs(self):
        for ifname, addrs in psutil.net_if_addrs().items():
            self.assertIsInstance(ifname, str)
            for addr in addrs:
                if enum is not None and (not PYPY):
                    self.assertIsInstance(addr.family, enum.IntEnum)
                else:
                    self.assertIsInstance(addr.family, int)
                self.assertIsInstance(addr.address, str)
                self.assertIsInstance(addr.netmask, (str, type(None)))
                self.assertIsInstance(addr.broadcast, (str, type(None)))

    def test_net_if_stats(self):
        for ifname, info in psutil.net_if_stats().items():
            self.assertIsInstance(ifname, str)
            self.assertIsInstance(info.isup, bool)
            if enum is not None:
                self.assertIsInstance(info.duplex, enum.IntEnum)
            else:
                self.assertIsInstance(info.duplex, int)
            self.assertIsInstance(info.speed, int)
            self.assertIsInstance(info.mtu, int)

    @unittest.skipIf(not HAS_NET_IO_COUNTERS, 'not supported')
    def test_net_io_counters(self):
        for ifname in psutil.net_io_counters(pernic=True):
            self.assertIsInstance(ifname, str)

    @unittest.skipIf(not HAS_SENSORS_FANS, 'not supported')
    def test_sensors_fans(self):
        for name, units in psutil.sensors_fans().items():
            self.assertIsInstance(name, str)
            for unit in units:
                self.assertIsInstance(unit.label, str)
                self.assertIsInstance(unit.current, (float, int, type(None)))

    @unittest.skipIf(not HAS_SENSORS_TEMPERATURES, 'not supported')
    def test_sensors_temperatures(self):
        for name, units in psutil.sensors_temperatures().items():
            self.assertIsInstance(name, str)
            for unit in units:
                self.assertIsInstance(unit.label, str)
                self.assertIsInstance(unit.current, (float, int, type(None)))
                self.assertIsInstance(unit.high, (float, int, type(None)))
                self.assertIsInstance(unit.critical, (float, int, type(None)))

    def test_boot_time(self):
        self.assertIsInstance(psutil.boot_time(), float)

    def test_users(self):
        for user in psutil.users():
            self.assertIsInstance(user.name, str)
            self.assertIsInstance(user.terminal, (str, type(None)))
            self.assertIsInstance(user.host, (str, type(None)))
            self.assertIsInstance(user.pid, (int, type(None)))