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
class TestAvailProcessAPIs(PsutilTestCase):

    def test_environ(self):
        self.assertEqual(hasattr(psutil.Process, 'environ'), LINUX or MACOS or WINDOWS or AIX or SUNOS or FREEBSD or OPENBSD or NETBSD)

    def test_uids(self):
        self.assertEqual(hasattr(psutil.Process, 'uids'), POSIX)

    def test_gids(self):
        self.assertEqual(hasattr(psutil.Process, 'uids'), POSIX)

    def test_terminal(self):
        self.assertEqual(hasattr(psutil.Process, 'terminal'), POSIX)

    def test_ionice(self):
        self.assertEqual(hasattr(psutil.Process, 'ionice'), LINUX or WINDOWS)

    @unittest.skipIf(GITHUB_ACTIONS and LINUX, 'unsupported on GITHUB_ACTIONS + LINUX')
    def test_rlimit(self):
        self.assertEqual(hasattr(psutil.Process, 'rlimit'), LINUX or FREEBSD)

    def test_io_counters(self):
        hasit = hasattr(psutil.Process, 'io_counters')
        self.assertEqual(hasit, not (MACOS or SUNOS))

    def test_num_fds(self):
        self.assertEqual(hasattr(psutil.Process, 'num_fds'), POSIX)

    def test_num_handles(self):
        self.assertEqual(hasattr(psutil.Process, 'num_handles'), WINDOWS)

    def test_cpu_affinity(self):
        self.assertEqual(hasattr(psutil.Process, 'cpu_affinity'), LINUX or WINDOWS or FREEBSD)

    def test_cpu_num(self):
        self.assertEqual(hasattr(psutil.Process, 'cpu_num'), LINUX or FREEBSD or SUNOS)

    def test_memory_maps(self):
        hasit = hasattr(psutil.Process, 'memory_maps')
        self.assertEqual(hasit, not (OPENBSD or NETBSD or AIX or MACOS))