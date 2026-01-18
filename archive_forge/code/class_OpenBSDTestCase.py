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
@unittest.skipIf(not OPENBSD, 'OPENBSD only')
class OpenBSDTestCase(PsutilTestCase):

    def test_boot_time(self):
        s = sysctl('kern.boottime')
        sys_bt = datetime.datetime.strptime(s, '%a %b %d %H:%M:%S %Y')
        psutil_bt = datetime.datetime.fromtimestamp(psutil.boot_time())
        self.assertEqual(sys_bt, psutil_bt)