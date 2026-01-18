import datetime
import os
import re
import time
import unittest
import psutil
from psutil import BSD
from psutil import FREEBSD
from psutil import NETBSD
from psutil import OPENBSD
from psutil.tests import HAS_BATTERY
from psutil.tests import TOLERANCE_SYS_MEM
from psutil.tests import PsutilTestCase
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import spawn_testproc
from psutil.tests import terminate
from psutil.tests import which
@unittest.skipIf(not FREEBSD, 'FREEBSD only')
class FreeBSDSystemTestCase(PsutilTestCase):

    @staticmethod
    def parse_swapinfo():
        output = sh('swapinfo -k').splitlines()[-1]
        parts = re.split('\\s+', output)
        if not parts:
            raise ValueError("Can't parse swapinfo: %s" % output)
        total, used, free = (int(p) * 1024 for p in parts[1:4])
        return (total, used, free)

    def test_cpu_frequency_against_sysctl(self):
        sensor = 'dev.cpu.0.freq'
        try:
            sysctl_result = int(sysctl(sensor))
        except RuntimeError:
            self.skipTest('frequencies not supported by kernel')
        self.assertEqual(psutil.cpu_freq().current, sysctl_result)
        sensor = 'dev.cpu.0.freq_levels'
        sysctl_result = sysctl(sensor)
        max_freq = int(sysctl_result.split()[0].split('/')[0])
        min_freq = int(sysctl_result.split()[-1].split('/')[0])
        self.assertEqual(psutil.cpu_freq().max, max_freq)
        self.assertEqual(psutil.cpu_freq().min, min_freq)

    @retry_on_failure()
    def test_vmem_active(self):
        syst = sysctl('vm.stats.vm.v_active_count') * PAGESIZE
        self.assertAlmostEqual(psutil.virtual_memory().active, syst, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_vmem_inactive(self):
        syst = sysctl('vm.stats.vm.v_inactive_count') * PAGESIZE
        self.assertAlmostEqual(psutil.virtual_memory().inactive, syst, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_vmem_wired(self):
        syst = sysctl('vm.stats.vm.v_wire_count') * PAGESIZE
        self.assertAlmostEqual(psutil.virtual_memory().wired, syst, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_vmem_cached(self):
        syst = sysctl('vm.stats.vm.v_cache_count') * PAGESIZE
        self.assertAlmostEqual(psutil.virtual_memory().cached, syst, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_vmem_free(self):
        syst = sysctl('vm.stats.vm.v_free_count') * PAGESIZE
        self.assertAlmostEqual(psutil.virtual_memory().free, syst, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_vmem_buffers(self):
        syst = sysctl('vfs.bufspace')
        self.assertAlmostEqual(psutil.virtual_memory().buffers, syst, delta=TOLERANCE_SYS_MEM)

    @unittest.skipIf(not MUSE_AVAILABLE, 'muse not installed')
    def test_muse_vmem_total(self):
        num = muse('Total')
        self.assertEqual(psutil.virtual_memory().total, num)

    @unittest.skipIf(not MUSE_AVAILABLE, 'muse not installed')
    @retry_on_failure()
    def test_muse_vmem_active(self):
        num = muse('Active')
        self.assertAlmostEqual(psutil.virtual_memory().active, num, delta=TOLERANCE_SYS_MEM)

    @unittest.skipIf(not MUSE_AVAILABLE, 'muse not installed')
    @retry_on_failure()
    def test_muse_vmem_inactive(self):
        num = muse('Inactive')
        self.assertAlmostEqual(psutil.virtual_memory().inactive, num, delta=TOLERANCE_SYS_MEM)

    @unittest.skipIf(not MUSE_AVAILABLE, 'muse not installed')
    @retry_on_failure()
    def test_muse_vmem_wired(self):
        num = muse('Wired')
        self.assertAlmostEqual(psutil.virtual_memory().wired, num, delta=TOLERANCE_SYS_MEM)

    @unittest.skipIf(not MUSE_AVAILABLE, 'muse not installed')
    @retry_on_failure()
    def test_muse_vmem_cached(self):
        num = muse('Cache')
        self.assertAlmostEqual(psutil.virtual_memory().cached, num, delta=TOLERANCE_SYS_MEM)

    @unittest.skipIf(not MUSE_AVAILABLE, 'muse not installed')
    @retry_on_failure()
    def test_muse_vmem_free(self):
        num = muse('Free')
        self.assertAlmostEqual(psutil.virtual_memory().free, num, delta=TOLERANCE_SYS_MEM)

    @unittest.skipIf(not MUSE_AVAILABLE, 'muse not installed')
    @retry_on_failure()
    def test_muse_vmem_buffers(self):
        num = muse('Buffer')
        self.assertAlmostEqual(psutil.virtual_memory().buffers, num, delta=TOLERANCE_SYS_MEM)

    def test_cpu_stats_ctx_switches(self):
        self.assertAlmostEqual(psutil.cpu_stats().ctx_switches, sysctl('vm.stats.sys.v_swtch'), delta=1000)

    def test_cpu_stats_interrupts(self):
        self.assertAlmostEqual(psutil.cpu_stats().interrupts, sysctl('vm.stats.sys.v_intr'), delta=1000)

    def test_cpu_stats_soft_interrupts(self):
        self.assertAlmostEqual(psutil.cpu_stats().soft_interrupts, sysctl('vm.stats.sys.v_soft'), delta=1000)

    @retry_on_failure()
    def test_cpu_stats_syscalls(self):
        self.assertAlmostEqual(psutil.cpu_stats().syscalls, sysctl('vm.stats.sys.v_syscall'), delta=200000)

    def test_swapmem_free(self):
        total, used, free = self.parse_swapinfo()
        self.assertAlmostEqual(psutil.swap_memory().free, free, delta=TOLERANCE_SYS_MEM)

    def test_swapmem_used(self):
        total, used, free = self.parse_swapinfo()
        self.assertAlmostEqual(psutil.swap_memory().used, used, delta=TOLERANCE_SYS_MEM)

    def test_swapmem_total(self):
        total, used, free = self.parse_swapinfo()
        self.assertAlmostEqual(psutil.swap_memory().total, total, delta=TOLERANCE_SYS_MEM)

    def test_boot_time(self):
        s = sysctl('sysctl kern.boottime')
        s = s[s.find(' sec = ') + 7:]
        s = s[:s.find(',')]
        btime = int(s)
        self.assertEqual(btime, psutil.boot_time())

    @unittest.skipIf(not HAS_BATTERY, 'no battery')
    def test_sensors_battery(self):

        def secs2hours(secs):
            m, s = divmod(secs, 60)
            h, m = divmod(m, 60)
            return '%d:%02d' % (h, m)
        out = sh('acpiconf -i 0')
        fields = dict([(x.split('\t')[0], x.split('\t')[-1]) for x in out.split('\n')])
        metrics = psutil.sensors_battery()
        percent = int(fields['Remaining capacity:'].replace('%', ''))
        remaining_time = fields['Remaining time:']
        self.assertEqual(metrics.percent, percent)
        if remaining_time == 'unknown':
            self.assertEqual(metrics.secsleft, psutil.POWER_TIME_UNLIMITED)
        else:
            self.assertEqual(secs2hours(metrics.secsleft), remaining_time)

    @unittest.skipIf(not HAS_BATTERY, 'no battery')
    def test_sensors_battery_against_sysctl(self):
        self.assertEqual(psutil.sensors_battery().percent, sysctl('hw.acpi.battery.life'))
        self.assertEqual(psutil.sensors_battery().power_plugged, sysctl('hw.acpi.acline') == 1)
        secsleft = psutil.sensors_battery().secsleft
        if secsleft < 0:
            self.assertEqual(sysctl('hw.acpi.battery.time'), -1)
        else:
            self.assertEqual(secsleft, sysctl('hw.acpi.battery.time') * 60)

    @unittest.skipIf(HAS_BATTERY, 'has battery')
    def test_sensors_battery_no_battery(self):
        with self.assertRaises(RuntimeError):
            sysctl('hw.acpi.battery.life')
            sysctl('hw.acpi.battery.time')
            sysctl('hw.acpi.acline')
        self.assertIsNone(psutil.sensors_battery())

    def test_sensors_temperatures_against_sysctl(self):
        num_cpus = psutil.cpu_count(True)
        for cpu in range(num_cpus):
            sensor = 'dev.cpu.%s.temperature' % cpu
            try:
                sysctl_result = int(float(sysctl(sensor)[:-1]))
            except RuntimeError:
                self.skipTest('temperatures not supported by kernel')
            self.assertAlmostEqual(psutil.sensors_temperatures()['coretemp'][cpu].current, sysctl_result, delta=10)
            sensor = 'dev.cpu.%s.coretemp.tjmax' % cpu
            sysctl_result = int(float(sysctl(sensor)[:-1]))
            self.assertEqual(psutil.sensors_temperatures()['coretemp'][cpu].high, sysctl_result)