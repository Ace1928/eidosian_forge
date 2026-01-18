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
@unittest.skipIf(PYPY, 'unreliable on PYPY')
class TestMemoryLeak(PsutilTestCase):
    """Test framework class for detecting function memory leaks,
    typically functions implemented in C which forgot to free() memory
    from the heap. It does so by checking whether the process memory
    usage increased before and after calling the function many times.

    Note that this is hard (probably impossible) to do reliably, due
    to how the OS handles memory, the GC and so on (memory can even
    decrease!). In order to avoid false positives, in case of failure
    (mem > 0) we retry the test for up to 5 times, increasing call
    repetitions each time. If the memory keeps increasing then it's a
    failure.

    If available (Linux, OSX, Windows), USS memory is used for comparison,
    since it's supposed to be more precise, see:
    https://gmpy.dev/blog/2016/real-process-memory-and-environ-in-python
    If not, RSS memory is used. mallinfo() on Linux and _heapwalk() on
    Windows may give even more precision, but at the moment are not
    implemented.

    PyPy appears to be completely unstable for this framework, probably
    because of its JIT, so tests on PYPY are skipped.

    Usage:

        class TestLeaks(psutil.tests.TestMemoryLeak):

            def test_fun(self):
                self.execute(some_function)
    """
    times = 200
    warmup_times = 10
    tolerance = 0
    retries = 10 if CI_TESTING else 5
    verbose = True
    _thisproc = psutil.Process()
    _psutil_debug_orig = bool(os.getenv('PSUTIL_DEBUG'))

    @classmethod
    def setUpClass(cls):
        psutil._set_debug(False)

    @classmethod
    def tearDownClass(cls):
        psutil._set_debug(cls._psutil_debug_orig)

    def _get_mem(self):
        mem = self._thisproc.memory_full_info()
        return getattr(mem, 'uss', mem.rss)

    def _get_num_fds(self):
        if POSIX:
            return self._thisproc.num_fds()
        else:
            return self._thisproc.num_handles()

    def _log(self, msg):
        if self.verbose:
            print_color(msg, color='yellow', file=sys.stderr)

    def _check_fds(self, fun):
        """Makes sure num_fds() (POSIX) or num_handles() (Windows) does
        not increase after calling a function.  Used to discover forgotten
        close(2) and CloseHandle syscalls.
        """
        before = self._get_num_fds()
        self.call(fun)
        after = self._get_num_fds()
        diff = after - before
        if diff < 0:
            raise self.fail('negative diff %r (gc probably collected a resource from a previous test)' % diff)
        if diff > 0:
            type_ = 'fd' if POSIX else 'handle'
            if diff > 1:
                type_ += 's'
            msg = '%s unclosed %s after calling %r' % (diff, type_, fun)
            raise self.fail(msg)

    def _call_ntimes(self, fun, times):
        """Get 2 distinct memory samples, before and after having
        called fun repeatedly, and return the memory difference.
        """
        gc.collect(generation=1)
        mem1 = self._get_mem()
        for x in range(times):
            ret = self.call(fun)
            del x, ret
        gc.collect(generation=1)
        mem2 = self._get_mem()
        self.assertEqual(gc.garbage, [])
        diff = mem2 - mem1
        return diff

    def _check_mem(self, fun, times, retries, tolerance):
        messages = []
        prev_mem = 0
        increase = times
        for idx in range(1, retries + 1):
            mem = self._call_ntimes(fun, times)
            msg = 'Run #%s: extra-mem=%s, per-call=%s, calls=%s' % (idx, bytes2human(mem), bytes2human(mem / times), times)
            messages.append(msg)
            success = mem <= tolerance or mem <= prev_mem
            if success:
                if idx > 1:
                    self._log(msg)
                return
            else:
                if idx == 1:
                    print()
                self._log(msg)
                times += increase
                prev_mem = mem
        raise self.fail('. '.join(messages))

    def call(self, fun):
        return fun()

    def execute(self, fun, times=None, warmup_times=None, retries=None, tolerance=None):
        """Test a callable."""
        times = times if times is not None else self.times
        warmup_times = warmup_times if warmup_times is not None else self.warmup_times
        retries = retries if retries is not None else self.retries
        tolerance = tolerance if tolerance is not None else self.tolerance
        try:
            assert times >= 1, 'times must be >= 1'
            assert warmup_times >= 0, 'warmup_times must be >= 0'
            assert retries >= 0, 'retries must be >= 0'
            assert tolerance >= 0, 'tolerance must be >= 0'
        except AssertionError as err:
            raise ValueError(str(err))
        self._call_ntimes(fun, warmup_times)
        self._check_fds(fun)
        self._check_mem(fun, times=times, retries=retries, tolerance=tolerance)

    def execute_w_exc(self, exc, fun, **kwargs):
        """Convenience method to test a callable while making sure it
        raises an exception on every call.
        """

        def call():
            self.assertRaises(exc, fun)
        self.execute(call, **kwargs)