import collections
import contextlib
import errno
import os
import socket
import stat
import subprocess
import unittest
import psutil
import psutil.tests
from psutil import FREEBSD
from psutil import NETBSD
from psutil import POSIX
from psutil._common import open_binary
from psutil._common import open_text
from psutil._common import supports_ipv6
from psutil.tests import CI_TESTING
from psutil.tests import COVERAGE
from psutil.tests import HAS_CONNECTIONS_UNIX
from psutil.tests import PYTHON_EXE
from psutil.tests import PYTHON_EXE_ENV
from psutil.tests import PsutilTestCase
from psutil.tests import TestMemoryLeak
from psutil.tests import bind_socket
from psutil.tests import bind_unix_socket
from psutil.tests import call_until
from psutil.tests import chdir
from psutil.tests import create_sockets
from psutil.tests import filter_proc_connections
from psutil.tests import get_free_port
from psutil.tests import is_namedtuple
from psutil.tests import mock
from psutil.tests import process_namespace
from psutil.tests import reap_children
from psutil.tests import retry
from psutil.tests import retry_on_failure
from psutil.tests import safe_mkdir
from psutil.tests import safe_rmpath
from psutil.tests import serialrun
from psutil.tests import system_namespace
from psutil.tests import tcp_socketpair
from psutil.tests import terminate
from psutil.tests import unix_socketpair
from psutil.tests import wait_for_file
from psutil.tests import wait_for_pid
class TestProcessUtils(PsutilTestCase):

    def test_reap_children(self):
        subp = self.spawn_testproc()
        p = psutil.Process(subp.pid)
        assert p.is_running()
        reap_children()
        assert not p.is_running()
        assert not psutil.tests._pids_started
        assert not psutil.tests._subprocesses_started

    def test_spawn_children_pair(self):
        child, grandchild = self.spawn_children_pair()
        self.assertNotEqual(child.pid, grandchild.pid)
        assert child.is_running()
        assert grandchild.is_running()
        children = psutil.Process().children()
        self.assertEqual(children, [child])
        children = psutil.Process().children(recursive=True)
        self.assertEqual(len(children), 2)
        self.assertIn(child, children)
        self.assertIn(grandchild, children)
        self.assertEqual(child.ppid(), os.getpid())
        self.assertEqual(grandchild.ppid(), child.pid)
        terminate(child)
        assert not child.is_running()
        assert grandchild.is_running()
        terminate(grandchild)
        assert not grandchild.is_running()

    @unittest.skipIf(not POSIX, 'POSIX only')
    def test_spawn_zombie(self):
        parent, zombie = self.spawn_zombie()
        self.assertEqual(zombie.status(), psutil.STATUS_ZOMBIE)

    def test_terminate(self):
        p = self.spawn_testproc()
        terminate(p)
        self.assertPidGone(p.pid)
        terminate(p)
        p = psutil.Process(self.spawn_testproc().pid)
        terminate(p)
        self.assertPidGone(p.pid)
        terminate(p)
        cmd = [PYTHON_EXE, '-c', 'import time; time.sleep(60);']
        p = psutil.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=PYTHON_EXE_ENV)
        terminate(p)
        self.assertPidGone(p.pid)
        terminate(p)
        pid = self.spawn_testproc().pid
        terminate(pid)
        self.assertPidGone(p.pid)
        terminate(pid)
        if POSIX:
            parent, zombie = self.spawn_zombie()
            terminate(parent)
            terminate(zombie)
            self.assertPidGone(parent.pid)
            self.assertPidGone(zombie.pid)