from __future__ import print_function
import atexit
import contextlib
import ctypes
import errno
import functools
import gc
import inspect
import os
import platform
import random
import re
import select
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import warnings
from socket import AF_INET
from socket import AF_INET6
from socket import SOCK_STREAM
import psutil
from psutil import AIX
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._common import bytes2human
from psutil._common import debug
from psutil._common import memoize
from psutil._common import print_color
from psutil._common import supports_ipv6
from psutil._compat import PY3
from psutil._compat import FileExistsError
from psutil._compat import FileNotFoundError
from psutil._compat import range
from psutil._compat import super
from psutil._compat import u
from psutil._compat import unicode
from psutil._compat import which
class PsutilTestCase(TestCase):
    """Test class providing auto-cleanup wrappers on top of process
    test utilities.
    """

    def get_testfn(self, suffix='', dir=None):
        fname = get_testfn(suffix=suffix, dir=dir)
        self.addCleanup(safe_rmpath, fname)
        return fname

    def spawn_testproc(self, *args, **kwds):
        sproc = spawn_testproc(*args, **kwds)
        self.addCleanup(terminate, sproc)
        return sproc

    def spawn_children_pair(self):
        child1, child2 = spawn_children_pair()
        self.addCleanup(terminate, child2)
        self.addCleanup(terminate, child1)
        return (child1, child2)

    def spawn_zombie(self):
        parent, zombie = spawn_zombie()
        self.addCleanup(terminate, zombie)
        self.addCleanup(terminate, parent)
        return (parent, zombie)

    def pyrun(self, *args, **kwds):
        sproc, srcfile = pyrun(*args, **kwds)
        self.addCleanup(safe_rmpath, srcfile)
        self.addCleanup(terminate, sproc)
        return sproc

    def _check_proc_exc(self, proc, exc):
        self.assertIsInstance(exc, psutil.Error)
        self.assertEqual(exc.pid, proc.pid)
        self.assertEqual(exc.name, proc._name)
        if exc.name:
            self.assertNotEqual(exc.name, '')
        if isinstance(exc, psutil.ZombieProcess):
            self.assertEqual(exc.ppid, proc._ppid)
            if exc.ppid is not None:
                self.assertGreaterEqual(exc.ppid, 0)
        str(exc)
        repr(exc)

    def assertPidGone(self, pid):
        with self.assertRaises(psutil.NoSuchProcess) as cm:
            try:
                psutil.Process(pid)
            except psutil.ZombieProcess:
                raise AssertionError("wasn't supposed to raise ZombieProcess")
        self.assertEqual(cm.exception.pid, pid)
        self.assertEqual(cm.exception.name, None)
        assert not psutil.pid_exists(pid), pid
        self.assertNotIn(pid, psutil.pids())
        self.assertNotIn(pid, [x.pid for x in psutil.process_iter()])

    def assertProcessGone(self, proc):
        self.assertPidGone(proc.pid)
        ns = process_namespace(proc)
        for fun, name in ns.iter(ns.all, clear_cache=True):
            with self.subTest(proc=proc, name=name):
                try:
                    ret = fun()
                except psutil.ZombieProcess:
                    raise
                except psutil.NoSuchProcess as exc:
                    self._check_proc_exc(proc, exc)
                else:
                    msg = "Process.%s() didn't raise NSP and returned %r" % (name, ret)
                    raise AssertionError(msg)
        proc.wait(timeout=0)

    def assertProcessZombie(self, proc):
        clone = psutil.Process(proc.pid)
        self.assertEqual(proc, clone)
        if not (OPENBSD or NETBSD):
            self.assertEqual(hash(proc), hash(clone))
        self.assertEqual(proc.status(), psutil.STATUS_ZOMBIE)
        assert proc.is_running()
        assert psutil.pid_exists(proc.pid)
        proc.as_dict()
        self.assertIn(proc.pid, psutil.pids())
        self.assertIn(proc.pid, [x.pid for x in psutil.process_iter()])
        psutil._pmap = {}
        self.assertIn(proc.pid, [x.pid for x in psutil.process_iter()])
        ns = process_namespace(proc)
        for fun, name in ns.iter(ns.all, clear_cache=True):
            with self.subTest(proc=proc, name=name):
                try:
                    fun()
                except (psutil.ZombieProcess, psutil.AccessDenied) as exc:
                    self._check_proc_exc(proc, exc)
        if LINUX:
            with self.assertRaises(psutil.ZombieProcess) as cm:
                proc.cmdline()
            self._check_proc_exc(proc, cm.exception)
            with self.assertRaises(psutil.ZombieProcess) as cm:
                proc.exe()
            self._check_proc_exc(proc, cm.exception)
            with self.assertRaises(psutil.ZombieProcess) as cm:
                proc.memory_maps()
            self._check_proc_exc(proc, cm.exception)
        proc.suspend()
        proc.resume()
        proc.terminate()
        proc.kill()
        assert proc.is_running()
        assert psutil.pid_exists(proc.pid)
        self.assertIn(proc.pid, psutil.pids())
        self.assertIn(proc.pid, [x.pid for x in psutil.process_iter()])
        psutil._pmap = {}
        self.assertIn(proc.pid, [x.pid for x in psutil.process_iter()])