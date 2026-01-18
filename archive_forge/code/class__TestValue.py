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
class _TestValue(BaseTestCase):
    ALLOWED_TYPES = ('processes',)
    codes_values = [('i', 4343, 24234), ('d', 3.625, -4.25), ('h', -232, 234), ('q', 2 ** 33, 2 ** 34), ('c', latin('x'), latin('y'))]

    def setUp(self):
        if not HAS_SHAREDCTYPES:
            self.skipTest('requires multiprocess.sharedctypes')

    @classmethod
    def _test(cls, values):
        for sv, cv in zip(values, cls.codes_values):
            sv.value = cv[2]

    def test_value(self, raw=False):
        if raw:
            values = [self.RawValue(code, value) for code, value, _ in self.codes_values]
        else:
            values = [self.Value(code, value) for code, value, _ in self.codes_values]
        for sv, cv in zip(values, self.codes_values):
            self.assertEqual(sv.value, cv[1])
        proc = self.Process(target=self._test, args=(values,))
        proc.daemon = True
        proc.start()
        proc.join()
        for sv, cv in zip(values, self.codes_values):
            self.assertEqual(sv.value, cv[2])

    def test_rawvalue(self):
        self.test_value(raw=True)

    def test_getobj_getlock(self):
        val1 = self.Value('i', 5)
        lock1 = val1.get_lock()
        obj1 = val1.get_obj()
        val2 = self.Value('i', 5, lock=None)
        lock2 = val2.get_lock()
        obj2 = val2.get_obj()
        lock = self.Lock()
        val3 = self.Value('i', 5, lock=lock)
        lock3 = val3.get_lock()
        obj3 = val3.get_obj()
        self.assertEqual(lock, lock3)
        arr4 = self.Value('i', 5, lock=False)
        self.assertFalse(hasattr(arr4, 'get_lock'))
        self.assertFalse(hasattr(arr4, 'get_obj'))
        self.assertRaises(AttributeError, self.Value, 'i', 5, lock='navalue')
        arr5 = self.RawValue('i', 5)
        self.assertFalse(hasattr(arr5, 'get_lock'))
        self.assertFalse(hasattr(arr5, 'get_obj'))