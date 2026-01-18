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
class _TestRemoteManager(BaseTestCase):
    ALLOWED_TYPES = ('manager',)
    values = ['hello world', None, True, 2.25, 'hallå världen', 'привіт світ', b'hall\xe5 v\xe4rlden']
    result = values[:]

    @classmethod
    def _putter(cls, address, authkey):
        manager = QueueManager2(address=address, authkey=authkey, serializer=SERIALIZER)
        manager.connect()
        queue = manager.get_queue()
        queue.put(tuple(cls.values))

    def test_remote(self):
        authkey = os.urandom(32)
        manager = QueueManager(address=(socket_helper.HOST, 0), authkey=authkey, serializer=SERIALIZER)
        manager.start()
        self.addCleanup(manager.shutdown)
        p = self.Process(target=self._putter, args=(manager.address, authkey))
        p.daemon = True
        p.start()
        manager2 = QueueManager2(address=manager.address, authkey=authkey, serializer=SERIALIZER)
        manager2.connect()
        queue = manager2.get_queue()
        self.assertEqual(queue.get(), self.result)
        self.assertRaises(Exception, queue.put, time.sleep)
        del queue