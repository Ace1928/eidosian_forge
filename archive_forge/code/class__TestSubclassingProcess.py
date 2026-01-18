import unittest
import unittest.mock
import queue as pyqueue
import textwrap
import time
import io
import itertools
import sys
import os
import gc
import errno
import signal
import array
import socket
import random
import logging
import subprocess
import struct
import operator
import pickle #XXX: use dill?
import weakref
import warnings
import test.support
import test.support.script_helper
from test import support
from test.support import hashlib_helper
from test.support import import_helper
from test.support import os_helper
from test.support import socket_helper
from test.support import threading_helper
from test.support import warnings_helper
import_helper.import_module('multiprocess.synchronize')
import threading
import multiprocess as multiprocessing
import multiprocess.connection
import multiprocess.dummy
import multiprocess.heap
import multiprocess.managers
import multiprocess.pool
import multiprocess.queues
from multiprocess import util
from multiprocess.connection import wait
from multiprocess.managers import BaseManager, BaseProxy, RemoteError
class _TestSubclassingProcess(BaseTestCase):
    ALLOWED_TYPES = ('processes',)

    def test_subclassing(self):
        uppercaser = _UpperCaser()
        uppercaser.daemon = True
        uppercaser.start()
        self.assertEqual(uppercaser.submit('hello'), 'HELLO')
        self.assertEqual(uppercaser.submit('world'), 'WORLD')
        uppercaser.stop()
        uppercaser.join()

    def test_stderr_flush(self):
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        testfn = os_helper.TESTFN
        self.addCleanup(os_helper.unlink, testfn)
        proc = self.Process(target=self._test_stderr_flush, args=(testfn,))
        proc.start()
        proc.join()
        with open(testfn, encoding='utf-8') as f:
            err = f.read()
            self.assertIn('ZeroDivisionError', err)
            self.assertIn('__init__.py', err)

    @classmethod
    def _test_stderr_flush(cls, testfn):
        fd = os.open(testfn, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        sys.stderr = open(fd, 'w', encoding='utf-8', closefd=False)
        1 / 0

    @classmethod
    def _test_sys_exit(cls, reason, testfn):
        fd = os.open(testfn, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        sys.stderr = open(fd, 'w', encoding='utf-8', closefd=False)
        sys.exit(reason)

    def test_sys_exit(self):
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        testfn = os_helper.TESTFN
        self.addCleanup(os_helper.unlink, testfn)
        for reason in ([1, 2, 3], 'ignore this'):
            p = self.Process(target=self._test_sys_exit, args=(reason, testfn))
            p.daemon = True
            p.start()
            join_process(p)
            self.assertEqual(p.exitcode, 1)
            with open(testfn, encoding='utf-8') as f:
                content = f.read()
            self.assertEqual(content.rstrip(), str(reason))
            os.unlink(testfn)
        cases = [((True,), 1), ((False,), 0), ((8,), 8), ((None,), 0), ((), 0)]
        for args, expected in cases:
            with self.subTest(args=args):
                p = self.Process(target=sys.exit, args=args)
                p.daemon = True
                p.start()
                join_process(p)
                self.assertEqual(p.exitcode, expected)