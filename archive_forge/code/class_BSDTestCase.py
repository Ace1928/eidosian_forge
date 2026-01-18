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
@unittest.skipIf(not BSD, 'BSD only')
class BSDTestCase(PsutilTestCase):
    """Generic tests common to all BSD variants."""

    @classmethod
    def setUpClass(cls):
        cls.pid = spawn_testproc().pid

    @classmethod
    def tearDownClass(cls):
        terminate(cls.pid)

    @unittest.skipIf(NETBSD, "-o lstart doesn't work on NETBSD")
    def test_process_create_time(self):
        output = sh('ps -o lstart -p %s' % self.pid)
        start_ps = output.replace('STARTED', '').strip()
        start_psutil = psutil.Process(self.pid).create_time()
        start_psutil = time.strftime('%a %b %e %H:%M:%S %Y', time.localtime(start_psutil))
        self.assertEqual(start_ps, start_psutil)

    def test_disks(self):

        def df(path):
            out = sh('df -k "%s"' % path).strip()
            lines = out.split('\n')
            lines.pop(0)
            line = lines.pop(0)
            dev, total, used, free = line.split()[:4]
            if dev == 'none':
                dev = ''
            total = int(total) * 1024
            used = int(used) * 1024
            free = int(free) * 1024
            return (dev, total, used, free)
        for part in psutil.disk_partitions(all=False):
            usage = psutil.disk_usage(part.mountpoint)
            dev, total, used, free = df(part.mountpoint)
            self.assertEqual(part.device, dev)
            self.assertEqual(usage.total, total)
            if abs(usage.free - free) > 10 * 1024 * 1024:
                raise self.fail('psutil=%s, df=%s' % (usage.free, free))
            if abs(usage.used - used) > 10 * 1024 * 1024:
                raise self.fail('psutil=%s, df=%s' % (usage.used, used))

    @unittest.skipIf(not which('sysctl'), 'sysctl cmd not available')
    def test_cpu_count_logical(self):
        syst = sysctl('hw.ncpu')
        self.assertEqual(psutil.cpu_count(logical=True), syst)

    @unittest.skipIf(not which('sysctl'), 'sysctl cmd not available')
    @unittest.skipIf(NETBSD, 'skipped on NETBSD')
    def test_virtual_memory_total(self):
        num = sysctl('hw.physmem')
        self.assertEqual(num, psutil.virtual_memory().total)

    @unittest.skipIf(not which('ifconfig'), 'ifconfig cmd not available')
    def test_net_if_stats(self):
        for name, stats in psutil.net_if_stats().items():
            try:
                out = sh('ifconfig %s' % name)
            except RuntimeError:
                pass
            else:
                self.assertEqual(stats.isup, 'RUNNING' in out, msg=out)
                if 'mtu' in out:
                    self.assertEqual(stats.mtu, int(re.findall('mtu (\\d+)', out)[0]))