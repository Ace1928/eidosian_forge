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
class TestWait(unittest.TestCase):

    @classmethod
    def _child_test_wait(cls, w, slow):
        for i in range(10):
            if slow:
                time.sleep(random.random() * 0.1)
            w.send((i, os.getpid()))
        w.close()

    def test_wait(self, slow=False):
        from multiprocess.connection import wait
        readers = []
        procs = []
        messages = []
        for i in range(4):
            r, w = multiprocessing.Pipe(duplex=False)
            p = multiprocessing.Process(target=self._child_test_wait, args=(w, slow))
            p.daemon = True
            p.start()
            w.close()
            readers.append(r)
            procs.append(p)
            self.addCleanup(p.join)
        while readers:
            for r in wait(readers):
                try:
                    msg = r.recv()
                except EOFError:
                    readers.remove(r)
                    r.close()
                else:
                    messages.append(msg)
        messages.sort()
        expected = sorted(((i, p.pid) for i in range(10) for p in procs))
        self.assertEqual(messages, expected)

    @classmethod
    def _child_test_wait_socket(cls, address, slow):
        s = socket.socket()
        s.connect(address)
        for i in range(10):
            if slow:
                time.sleep(random.random() * 0.1)
            s.sendall(('%s\n' % i).encode('ascii'))
        s.close()

    def test_wait_socket(self, slow=False):
        from multiprocess.connection import wait
        l = socket.create_server((socket_helper.HOST, 0))
        addr = l.getsockname()
        readers = []
        procs = []
        dic = {}
        for i in range(4):
            p = multiprocessing.Process(target=self._child_test_wait_socket, args=(addr, slow))
            p.daemon = True
            p.start()
            procs.append(p)
            self.addCleanup(p.join)
        for i in range(4):
            r, _ = l.accept()
            readers.append(r)
            dic[r] = []
        l.close()
        while readers:
            for r in wait(readers):
                msg = r.recv(32)
                if not msg:
                    readers.remove(r)
                    r.close()
                else:
                    dic[r].append(msg)
        expected = ''.join(('%s\n' % i for i in range(10))).encode('ascii')
        for v in dic.values():
            self.assertEqual(b''.join(v), expected)

    def test_wait_slow(self):
        self.test_wait(True)

    def test_wait_socket_slow(self):
        self.test_wait_socket(True)

    def test_wait_timeout(self):
        from multiprocess.connection import wait
        expected = 5
        a, b = multiprocessing.Pipe()
        start = time.monotonic()
        res = wait([a, b], expected)
        delta = time.monotonic() - start
        self.assertEqual(res, [])
        self.assertLess(delta, expected * 2)
        self.assertGreater(delta, expected * 0.5)
        b.send(None)
        start = time.monotonic()
        res = wait([a, b], 20)
        delta = time.monotonic() - start
        self.assertEqual(res, [a])
        self.assertLess(delta, 0.4)

    @classmethod
    def signal_and_sleep(cls, sem, period):
        sem.release()
        time.sleep(period)

    def test_wait_integer(self):
        from multiprocess.connection import wait
        expected = 3
        sorted_ = lambda l: sorted(l, key=lambda x: id(x))
        sem = multiprocessing.Semaphore(0)
        a, b = multiprocessing.Pipe()
        p = multiprocessing.Process(target=self.signal_and_sleep, args=(sem, expected))
        p.start()
        self.assertIsInstance(p.sentinel, int)
        self.assertTrue(sem.acquire(timeout=20))
        start = time.monotonic()
        res = wait([a, p.sentinel, b], expected + 20)
        delta = time.monotonic() - start
        self.assertEqual(res, [p.sentinel])
        self.assertLess(delta, expected + 2)
        self.assertGreater(delta, expected - 2)
        a.send(None)
        start = time.monotonic()
        res = wait([a, p.sentinel, b], 20)
        delta = time.monotonic() - start
        self.assertEqual(sorted_(res), sorted_([p.sentinel, b]))
        self.assertLess(delta, 0.4)
        b.send(None)
        start = time.monotonic()
        res = wait([a, p.sentinel, b], 20)
        delta = time.monotonic() - start
        self.assertEqual(sorted_(res), sorted_([a, p.sentinel, b]))
        self.assertLess(delta, 0.4)
        p.terminate()
        p.join()

    def test_neg_timeout(self):
        from multiprocess.connection import wait
        a, b = multiprocessing.Pipe()
        t = time.monotonic()
        res = wait([a], timeout=-1)
        t = time.monotonic() - t
        self.assertEqual(res, [])
        self.assertLess(t, 1)
        a.close()
        b.close()