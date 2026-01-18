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
class TestSimpleQueue(unittest.TestCase):

    @classmethod
    def _test_empty(cls, queue, child_can_start, parent_can_continue):
        child_can_start.wait()
        try:
            queue.put(queue.empty())
            queue.put(queue.empty())
        finally:
            parent_can_continue.set()

    def test_empty(self):
        queue = multiprocessing.SimpleQueue()
        child_can_start = multiprocessing.Event()
        parent_can_continue = multiprocessing.Event()
        proc = multiprocessing.Process(target=self._test_empty, args=(queue, child_can_start, parent_can_continue))
        proc.daemon = True
        proc.start()
        self.assertTrue(queue.empty())
        child_can_start.set()
        parent_can_continue.wait()
        self.assertFalse(queue.empty())
        self.assertEqual(queue.get(), True)
        self.assertEqual(queue.get(), False)
        self.assertTrue(queue.empty())
        proc.join()

    def test_close(self):
        queue = multiprocessing.SimpleQueue()
        queue.close()
        queue.close()

    @test.support.cpython_only
    def test_closed(self):
        queue = multiprocessing.SimpleQueue()
        queue.close()
        self.assertTrue(queue._reader.closed)
        self.assertTrue(queue._writer.closed)