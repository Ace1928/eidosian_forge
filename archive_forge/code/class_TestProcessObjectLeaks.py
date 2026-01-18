from __future__ import print_function
import functools
import os
import platform
import unittest
import psutil
import psutil._common
from psutil import LINUX
from psutil import MACOS
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._compat import ProcessLookupError
from psutil._compat import super
from psutil.tests import HAS_CPU_AFFINITY
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_ENVIRON
from psutil.tests import HAS_IONICE
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_PROC_CPU_NUM
from psutil.tests import HAS_PROC_IO_COUNTERS
from psutil.tests import HAS_RLIMIT
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import TestMemoryLeak
from psutil.tests import create_sockets
from psutil.tests import get_testfn
from psutil.tests import process_namespace
from psutil.tests import skip_on_access_denied
from psutil.tests import spawn_testproc
from psutil.tests import system_namespace
from psutil.tests import terminate
class TestProcessObjectLeaks(TestMemoryLeak):
    """Test leaks of Process class methods."""
    proc = thisproc

    def test_coverage(self):
        ns = process_namespace(None)
        ns.test_class_coverage(self, ns.getters + ns.setters)

    @fewtimes_if_linux()
    def test_name(self):
        self.execute(self.proc.name)

    @fewtimes_if_linux()
    def test_cmdline(self):
        self.execute(self.proc.cmdline)

    @fewtimes_if_linux()
    def test_exe(self):
        self.execute(self.proc.exe)

    @fewtimes_if_linux()
    def test_ppid(self):
        self.execute(self.proc.ppid)

    @unittest.skipIf(not POSIX, 'POSIX only')
    @fewtimes_if_linux()
    def test_uids(self):
        self.execute(self.proc.uids)

    @unittest.skipIf(not POSIX, 'POSIX only')
    @fewtimes_if_linux()
    def test_gids(self):
        self.execute(self.proc.gids)

    @fewtimes_if_linux()
    def test_status(self):
        self.execute(self.proc.status)

    def test_nice(self):
        self.execute(self.proc.nice)

    def test_nice_set(self):
        niceness = thisproc.nice()
        self.execute(lambda: self.proc.nice(niceness))

    @unittest.skipIf(not HAS_IONICE, 'not supported')
    def test_ionice(self):
        self.execute(self.proc.ionice)

    @unittest.skipIf(not HAS_IONICE, 'not supported')
    def test_ionice_set(self):
        if WINDOWS:
            value = thisproc.ionice()
            self.execute(lambda: self.proc.ionice(value))
        else:
            self.execute(lambda: self.proc.ionice(psutil.IOPRIO_CLASS_NONE))
            fun = functools.partial(cext.proc_ioprio_set, os.getpid(), -1, 0)
            self.execute_w_exc(OSError, fun)

    @unittest.skipIf(not HAS_PROC_IO_COUNTERS, 'not supported')
    @fewtimes_if_linux()
    def test_io_counters(self):
        self.execute(self.proc.io_counters)

    @unittest.skipIf(POSIX, 'worthless on POSIX')
    def test_username(self):
        psutil.Process().username()
        self.execute(self.proc.username)

    @fewtimes_if_linux()
    def test_create_time(self):
        self.execute(self.proc.create_time)

    @fewtimes_if_linux()
    @skip_on_access_denied(only_if=OPENBSD)
    def test_num_threads(self):
        self.execute(self.proc.num_threads)

    @unittest.skipIf(not WINDOWS, 'WINDOWS only')
    def test_num_handles(self):
        self.execute(self.proc.num_handles)

    @unittest.skipIf(not POSIX, 'POSIX only')
    @fewtimes_if_linux()
    def test_num_fds(self):
        self.execute(self.proc.num_fds)

    @fewtimes_if_linux()
    def test_num_ctx_switches(self):
        self.execute(self.proc.num_ctx_switches)

    @fewtimes_if_linux()
    @skip_on_access_denied(only_if=OPENBSD)
    def test_threads(self):
        self.execute(self.proc.threads)

    @fewtimes_if_linux()
    def test_cpu_times(self):
        self.execute(self.proc.cpu_times)

    @fewtimes_if_linux()
    @unittest.skipIf(not HAS_PROC_CPU_NUM, 'not supported')
    def test_cpu_num(self):
        self.execute(self.proc.cpu_num)

    @fewtimes_if_linux()
    def test_memory_info(self):
        self.execute(self.proc.memory_info)

    @fewtimes_if_linux()
    def test_memory_full_info(self):
        self.execute(self.proc.memory_full_info)

    @unittest.skipIf(not POSIX, 'POSIX only')
    @fewtimes_if_linux()
    def test_terminal(self):
        self.execute(self.proc.terminal)

    def test_resume(self):
        times = FEW_TIMES if POSIX else self.times
        self.execute(self.proc.resume, times=times)

    @fewtimes_if_linux()
    def test_cwd(self):
        self.execute(self.proc.cwd)

    @unittest.skipIf(not HAS_CPU_AFFINITY, 'not supported')
    def test_cpu_affinity(self):
        self.execute(self.proc.cpu_affinity)

    @unittest.skipIf(not HAS_CPU_AFFINITY, 'not supported')
    def test_cpu_affinity_set(self):
        affinity = thisproc.cpu_affinity()
        self.execute(lambda: self.proc.cpu_affinity(affinity))
        self.execute_w_exc(ValueError, lambda: self.proc.cpu_affinity([-1]))

    @fewtimes_if_linux()
    def test_open_files(self):
        with open(get_testfn(), 'w'):
            self.execute(self.proc.open_files)

    @unittest.skipIf(not HAS_MEMORY_MAPS, 'not supported')
    @fewtimes_if_linux()
    def test_memory_maps(self):
        self.execute(self.proc.memory_maps)

    @unittest.skipIf(not LINUX, 'LINUX only')
    @unittest.skipIf(not HAS_RLIMIT, 'not supported')
    def test_rlimit(self):
        self.execute(lambda: self.proc.rlimit(psutil.RLIMIT_NOFILE))

    @unittest.skipIf(not LINUX, 'LINUX only')
    @unittest.skipIf(not HAS_RLIMIT, 'not supported')
    def test_rlimit_set(self):
        limit = thisproc.rlimit(psutil.RLIMIT_NOFILE)
        self.execute(lambda: self.proc.rlimit(psutil.RLIMIT_NOFILE, limit))
        self.execute_w_exc((OSError, ValueError), lambda: self.proc.rlimit(-1))

    @fewtimes_if_linux()
    @unittest.skipIf(WINDOWS, 'worthless on WINDOWS')
    def test_connections(self):
        with create_sockets():
            kind = 'inet' if SUNOS else 'all'
            self.execute(lambda: self.proc.connections(kind))

    @unittest.skipIf(not HAS_ENVIRON, 'not supported')
    def test_environ(self):
        self.execute(self.proc.environ)

    @unittest.skipIf(not WINDOWS, 'WINDOWS only')
    def test_proc_info(self):
        self.execute(lambda: cext.proc_info(os.getpid()))