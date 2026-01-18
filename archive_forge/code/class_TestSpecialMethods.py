import ast
import collections
import errno
import json
import os
import pickle
import socket
import stat
import unittest
import psutil
import psutil.tests
from psutil import LINUX
from psutil import POSIX
from psutil import WINDOWS
from psutil._common import bcat
from psutil._common import cat
from psutil._common import debug
from psutil._common import isfile_strict
from psutil._common import memoize
from psutil._common import memoize_when_activated
from psutil._common import parse_environ_block
from psutil._common import supports_ipv6
from psutil._common import wrap_numbers
from psutil._compat import PY3
from psutil._compat import FileNotFoundError
from psutil._compat import redirect_stderr
from psutil.tests import APPVEYOR
from psutil.tests import CI_TESTING
from psutil.tests import HAS_BATTERY
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import PYTHON_EXE
from psutil.tests import PYTHON_EXE_ENV
from psutil.tests import SCRIPTS_DIR
from psutil.tests import PsutilTestCase
from psutil.tests import mock
from psutil.tests import reload_module
from psutil.tests import sh
class TestSpecialMethods(PsutilTestCase):

    def test_check_pid_range(self):
        with self.assertRaises(OverflowError):
            psutil._psplatform.cext.check_pid_range(2 ** 128)
        with self.assertRaises(psutil.NoSuchProcess):
            psutil.Process(2 ** 128)

    def test_process__repr__(self, func=repr):
        p = psutil.Process(self.spawn_testproc().pid)
        r = func(p)
        self.assertIn('psutil.Process', r)
        self.assertIn('pid=%s' % p.pid, r)
        self.assertIn("name='%s'" % str(p.name()), r.replace("name=u'", "name='"))
        self.assertIn('status=', r)
        self.assertNotIn('exitcode=', r)
        p.terminate()
        p.wait()
        r = func(p)
        self.assertIn("status='terminated'", r)
        self.assertIn('exitcode=', r)
        with mock.patch.object(psutil.Process, 'name', side_effect=psutil.ZombieProcess(os.getpid())):
            p = psutil.Process()
            r = func(p)
            self.assertIn('pid=%s' % p.pid, r)
            self.assertIn("status='zombie'", r)
            self.assertNotIn('name=', r)
        with mock.patch.object(psutil.Process, 'name', side_effect=psutil.NoSuchProcess(os.getpid())):
            p = psutil.Process()
            r = func(p)
            self.assertIn('pid=%s' % p.pid, r)
            self.assertIn('terminated', r)
            self.assertNotIn('name=', r)
        with mock.patch.object(psutil.Process, 'name', side_effect=psutil.AccessDenied(os.getpid())):
            p = psutil.Process()
            r = func(p)
            self.assertIn('pid=%s' % p.pid, r)
            self.assertNotIn('name=', r)

    def test_process__str__(self):
        self.test_process__repr__(func=str)

    def test_error__repr__(self):
        self.assertEqual(repr(psutil.Error()), 'psutil.Error()')

    def test_error__str__(self):
        self.assertEqual(str(psutil.Error()), '')

    def test_no_such_process__repr__(self):
        self.assertEqual(repr(psutil.NoSuchProcess(321)), "psutil.NoSuchProcess(pid=321, msg='process no longer exists')")
        self.assertEqual(repr(psutil.NoSuchProcess(321, name='name', msg='msg')), "psutil.NoSuchProcess(pid=321, name='name', msg='msg')")

    def test_no_such_process__str__(self):
        self.assertEqual(str(psutil.NoSuchProcess(321)), 'process no longer exists (pid=321)')
        self.assertEqual(str(psutil.NoSuchProcess(321, name='name', msg='msg')), "msg (pid=321, name='name')")

    def test_zombie_process__repr__(self):
        self.assertEqual(repr(psutil.ZombieProcess(321)), 'psutil.ZombieProcess(pid=321, msg="PID still exists but it\'s a zombie")')
        self.assertEqual(repr(psutil.ZombieProcess(321, name='name', ppid=320, msg='foo')), "psutil.ZombieProcess(pid=321, ppid=320, name='name', msg='foo')")

    def test_zombie_process__str__(self):
        self.assertEqual(str(psutil.ZombieProcess(321)), "PID still exists but it's a zombie (pid=321)")
        self.assertEqual(str(psutil.ZombieProcess(321, name='name', ppid=320, msg='foo')), "foo (pid=321, ppid=320, name='name')")

    def test_access_denied__repr__(self):
        self.assertEqual(repr(psutil.AccessDenied(321)), 'psutil.AccessDenied(pid=321)')
        self.assertEqual(repr(psutil.AccessDenied(321, name='name', msg='msg')), "psutil.AccessDenied(pid=321, name='name', msg='msg')")

    def test_access_denied__str__(self):
        self.assertEqual(str(psutil.AccessDenied(321)), '(pid=321)')
        self.assertEqual(str(psutil.AccessDenied(321, name='name', msg='msg')), "msg (pid=321, name='name')")

    def test_timeout_expired__repr__(self):
        self.assertEqual(repr(psutil.TimeoutExpired(5)), "psutil.TimeoutExpired(seconds=5, msg='timeout after 5 seconds')")
        self.assertEqual(repr(psutil.TimeoutExpired(5, pid=321, name='name')), "psutil.TimeoutExpired(pid=321, name='name', seconds=5, msg='timeout after 5 seconds')")

    def test_timeout_expired__str__(self):
        self.assertEqual(str(psutil.TimeoutExpired(5)), 'timeout after 5 seconds')
        self.assertEqual(str(psutil.TimeoutExpired(5, pid=321, name='name')), "timeout after 5 seconds (pid=321, name='name')")

    def test_process__eq__(self):
        p1 = psutil.Process()
        p2 = psutil.Process()
        self.assertEqual(p1, p2)
        p2._ident = (0, 0)
        self.assertNotEqual(p1, p2)
        self.assertNotEqual(p1, 'foo')

    def test_process__hash__(self):
        s = set([psutil.Process(), psutil.Process()])
        self.assertEqual(len(s), 1)