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
class TestCloseFds(unittest.TestCase):

    def get_high_socket_fd(self):
        if WIN32:
            return socket.socket().detach()
        else:
            fd = socket.socket().detach()
            to_close = []
            while fd < 50:
                to_close.append(fd)
                fd = os.dup(fd)
            for x in to_close:
                os.close(x)
            return fd

    def close(self, fd):
        if WIN32:
            socket.socket(socket.AF_INET, socket.SOCK_STREAM, fileno=fd).close()
        else:
            os.close(fd)

    @classmethod
    def _test_closefds(cls, conn, fd):
        try:
            s = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM)
        except Exception as e:
            conn.send(e)
        else:
            s.close()
            conn.send(None)

    def test_closefd(self):
        if not HAS_REDUCTION:
            raise unittest.SkipTest('requires fd pickling')
        reader, writer = multiprocessing.Pipe()
        fd = self.get_high_socket_fd()
        try:
            p = multiprocessing.Process(target=self._test_closefds, args=(writer, fd))
            p.start()
            writer.close()
            e = reader.recv()
            join_process(p)
        finally:
            self.close(fd)
            writer.close()
            reader.close()
        if multiprocessing.get_start_method() == 'fork':
            self.assertIs(e, None)
        else:
            WSAENOTSOCK = 10038
            self.assertIsInstance(e, OSError)
            self.assertTrue(e.errno == errno.EBADF or e.winerror == WSAENOTSOCK, e)