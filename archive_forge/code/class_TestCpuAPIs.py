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
class TestCpuAPIs(PsutilTestCase):

    def test_cpu_count_logical(self):
        logical = psutil.cpu_count()
        self.assertIsNotNone(logical)
        self.assertEqual(logical, len(psutil.cpu_times(percpu=True)))
        self.assertGreaterEqual(logical, 1)
        if os.path.exists('/proc/cpuinfo'):
            with open('/proc/cpuinfo') as fd:
                cpuinfo_data = fd.read()
            if 'physical id' not in cpuinfo_data:
                raise unittest.SkipTest("cpuinfo doesn't include physical id")

    def test_cpu_count_cores(self):
        logical = psutil.cpu_count()
        cores = psutil.cpu_count(logical=False)
        if cores is None:
            raise self.skipTest('cpu_count_cores() is None')
        if WINDOWS and sys.getwindowsversion()[:2] <= (6, 1):
            self.assertIsNone(cores)
        else:
            self.assertGreaterEqual(cores, 1)
            self.assertGreaterEqual(logical, cores)

    def test_cpu_count_none(self):
        for val in (-1, 0, None):
            with mock.patch('psutil._psplatform.cpu_count_logical', return_value=val) as m:
                self.assertIsNone(psutil.cpu_count())
                assert m.called
            with mock.patch('psutil._psplatform.cpu_count_cores', return_value=val) as m:
                self.assertIsNone(psutil.cpu_count(logical=False))
                assert m.called

    def test_cpu_times(self):
        total = 0
        times = psutil.cpu_times()
        sum(times)
        for cp_time in times:
            self.assertIsInstance(cp_time, float)
            self.assertGreaterEqual(cp_time, 0.0)
            total += cp_time
        self.assertAlmostEqual(total, sum(times))
        str(times)

    def test_cpu_times_time_increases(self):
        t1 = sum(psutil.cpu_times())
        stop_at = time.time() + GLOBAL_TIMEOUT
        while time.time() < stop_at:
            t2 = sum(psutil.cpu_times())
            if t2 > t1:
                return
        raise self.fail('time remained the same')

    def test_per_cpu_times(self):
        for times in psutil.cpu_times(percpu=True):
            total = 0
            sum(times)
            for cp_time in times:
                self.assertIsInstance(cp_time, float)
                self.assertGreaterEqual(cp_time, 0.0)
                total += cp_time
            self.assertAlmostEqual(total, sum(times))
            str(times)
        self.assertEqual(len(psutil.cpu_times(percpu=True)[0]), len(psutil.cpu_times(percpu=False)))

    def test_per_cpu_times_2(self):
        tot1 = psutil.cpu_times(percpu=True)
        giveup_at = time.time() + GLOBAL_TIMEOUT
        while True:
            if time.time() >= giveup_at:
                return self.fail('timeout')
            tot2 = psutil.cpu_times(percpu=True)
            for t1, t2 in zip(tot1, tot2):
                t1, t2 = (psutil._cpu_busy_time(t1), psutil._cpu_busy_time(t2))
                difference = t2 - t1
                if difference >= 0.05:
                    return

    @unittest.skipIf(CI_TESTING and OPENBSD, 'unreliable on OPENBSD + CI')
    def test_cpu_times_comparison(self):
        base = psutil.cpu_times()
        per_cpu = psutil.cpu_times(percpu=True)
        summed_values = base._make([sum(num) for num in zip(*per_cpu)])
        for field in base._fields:
            with self.subTest(field=field, base=base, per_cpu=per_cpu):
                self.assertAlmostEqual(getattr(base, field), getattr(summed_values, field), delta=1)

    def _test_cpu_percent(self, percent, last_ret, new_ret):
        try:
            self.assertIsInstance(percent, float)
            self.assertGreaterEqual(percent, 0.0)
            self.assertIsNot(percent, -0.0)
            self.assertLessEqual(percent, 100.0 * psutil.cpu_count())
        except AssertionError as err:
            raise AssertionError('\n%s\nlast=%s\nnew=%s' % (err, pprint.pformat(last_ret), pprint.pformat(new_ret)))

    def test_cpu_percent(self):
        last = psutil.cpu_percent(interval=0.001)
        for _ in range(100):
            new = psutil.cpu_percent(interval=None)
            self._test_cpu_percent(new, last, new)
            last = new
        with self.assertRaises(ValueError):
            psutil.cpu_percent(interval=-1)

    def test_per_cpu_percent(self):
        last = psutil.cpu_percent(interval=0.001, percpu=True)
        self.assertEqual(len(last), psutil.cpu_count())
        for _ in range(100):
            new = psutil.cpu_percent(interval=None, percpu=True)
            for percent in new:
                self._test_cpu_percent(percent, last, new)
            last = new
        with self.assertRaises(ValueError):
            psutil.cpu_percent(interval=-1, percpu=True)

    def test_cpu_times_percent(self):
        last = psutil.cpu_times_percent(interval=0.001)
        for _ in range(100):
            new = psutil.cpu_times_percent(interval=None)
            for percent in new:
                self._test_cpu_percent(percent, last, new)
            self._test_cpu_percent(sum(new), last, new)
            last = new
        with self.assertRaises(ValueError):
            psutil.cpu_times_percent(interval=-1)

    def test_per_cpu_times_percent(self):
        last = psutil.cpu_times_percent(interval=0.001, percpu=True)
        self.assertEqual(len(last), psutil.cpu_count())
        for _ in range(100):
            new = psutil.cpu_times_percent(interval=None, percpu=True)
            for cpu in new:
                for percent in cpu:
                    self._test_cpu_percent(percent, last, new)
                self._test_cpu_percent(sum(cpu), last, new)
            last = new

    def test_per_cpu_times_percent_negative(self):
        psutil.cpu_times_percent(percpu=True)
        zero_times = [x._make([0 for x in range(len(x._fields))]) for x in psutil.cpu_times(percpu=True)]
        with mock.patch('psutil.cpu_times', return_value=zero_times):
            for cpu in psutil.cpu_times_percent(percpu=True):
                for percent in cpu:
                    self._test_cpu_percent(percent, None, None)

    def test_cpu_stats(self):
        infos = psutil.cpu_stats()
        self.assertEqual(infos._fields, ('ctx_switches', 'interrupts', 'soft_interrupts', 'syscalls'))
        for name in infos._fields:
            value = getattr(infos, name)
            self.assertGreaterEqual(value, 0)
            if not AIX and name in ('ctx_switches', 'interrupts'):
                self.assertGreater(value, 0)

    @unittest.skipIf(MACOS and platform.machine() == 'arm64', 'skipped due to #1892')
    @unittest.skipIf(not HAS_CPU_FREQ, 'not supported')
    def test_cpu_freq(self):

        def check_ls(ls):
            for nt in ls:
                self.assertEqual(nt._fields, ('current', 'min', 'max'))
                if nt.max != 0.0:
                    self.assertLessEqual(nt.current, nt.max)
                for name in nt._fields:
                    value = getattr(nt, name)
                    self.assertIsInstance(value, (int, long, float))
                    self.assertGreaterEqual(value, 0)
        ls = psutil.cpu_freq(percpu=True)
        if FREEBSD and (not ls):
            raise self.skipTest('returns empty list on FreeBSD')
        assert ls, ls
        check_ls([psutil.cpu_freq(percpu=False)])
        if LINUX:
            self.assertEqual(len(ls), psutil.cpu_count())

    @unittest.skipIf(not HAS_GETLOADAVG, 'not supported')
    def test_getloadavg(self):
        loadavg = psutil.getloadavg()
        self.assertEqual(len(loadavg), 3)
        for load in loadavg:
            self.assertIsInstance(load, float)
            self.assertGreaterEqual(load, 0.0)