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
class FreeBSDPsutilTestCase(PsutilTestCase):

    @classmethod
    def setUpClass(cls):
        cls.pid = spawn_testproc().pid

    @classmethod
    def tearDownClass(cls):
        terminate(cls.pid)

    @retry_on_failure()
    def test_memory_maps(self):
        out = sh('procstat -v %s' % self.pid)
        maps = psutil.Process(self.pid).memory_maps(grouped=False)
        lines = out.split('\n')[1:]
        while lines:
            line = lines.pop()
            fields = line.split()
            _, start, stop, perms, res = fields[:5]
            map = maps.pop()
            self.assertEqual('%s-%s' % (start, stop), map.addr)
            self.assertEqual(int(res), map.rss)
            if not map.path.startswith('['):
                self.assertEqual(fields[10], map.path)

    def test_exe(self):
        out = sh('procstat -b %s' % self.pid)
        self.assertEqual(psutil.Process(self.pid).exe(), out.split('\n')[1].split()[-1])

    def test_cmdline(self):
        out = sh('procstat -c %s' % self.pid)
        self.assertEqual(' '.join(psutil.Process(self.pid).cmdline()), ' '.join(out.split('\n')[1].split()[2:]))

    def test_uids_gids(self):
        out = sh('procstat -s %s' % self.pid)
        euid, ruid, suid, egid, rgid, sgid = out.split('\n')[1].split()[2:8]
        p = psutil.Process(self.pid)
        uids = p.uids()
        gids = p.gids()
        self.assertEqual(uids.real, int(ruid))
        self.assertEqual(uids.effective, int(euid))
        self.assertEqual(uids.saved, int(suid))
        self.assertEqual(gids.real, int(rgid))
        self.assertEqual(gids.effective, int(egid))
        self.assertEqual(gids.saved, int(sgid))

    @retry_on_failure()
    def test_ctx_switches(self):
        tested = []
        out = sh('procstat -r %s' % self.pid)
        p = psutil.Process(self.pid)
        for line in out.split('\n'):
            line = line.lower().strip()
            if ' voluntary context' in line:
                pstat_value = int(line.split()[-1])
                psutil_value = p.num_ctx_switches().voluntary
                self.assertEqual(pstat_value, psutil_value)
                tested.append(None)
            elif ' involuntary context' in line:
                pstat_value = int(line.split()[-1])
                psutil_value = p.num_ctx_switches().involuntary
                self.assertEqual(pstat_value, psutil_value)
                tested.append(None)
        if len(tested) != 2:
            raise RuntimeError("couldn't find lines match in procstat out")

    @retry_on_failure()
    def test_cpu_times(self):
        tested = []
        out = sh('procstat -r %s' % self.pid)
        p = psutil.Process(self.pid)
        for line in out.split('\n'):
            line = line.lower().strip()
            if 'user time' in line:
                pstat_value = float('0.' + line.split()[-1].split('.')[-1])
                psutil_value = p.cpu_times().user
                self.assertEqual(pstat_value, psutil_value)
                tested.append(None)
            elif 'system time' in line:
                pstat_value = float('0.' + line.split()[-1].split('.')[-1])
                psutil_value = p.cpu_times().system
                self.assertEqual(pstat_value, psutil_value)
                tested.append(None)
        if len(tested) != 2:
            raise RuntimeError("couldn't find lines match in procstat out")