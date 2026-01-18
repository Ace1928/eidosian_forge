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
class _TestMyManager(BaseTestCase):
    ALLOWED_TYPES = ('manager',)

    def test_mymanager(self):
        manager = MyManager()
        manager.start()
        self.common(manager)
        manager.shutdown()
        self.assertIn(manager._process.exitcode, (0, -signal.SIGTERM))

    def test_mymanager_context(self):
        with MyManager() as manager:
            self.common(manager)
        self.assertIn(manager._process.exitcode, (0, -signal.SIGTERM))

    def test_mymanager_context_prestarted(self):
        manager = MyManager()
        manager.start()
        with manager:
            self.common(manager)
        self.assertEqual(manager._process.exitcode, 0)

    def common(self, manager):
        foo = manager.Foo()
        bar = manager.Bar()
        baz = manager.baz()
        foo_methods = [name for name in ('f', 'g', '_h') if hasattr(foo, name)]
        bar_methods = [name for name in ('f', 'g', '_h') if hasattr(bar, name)]
        self.assertEqual(foo_methods, ['f', 'g'])
        self.assertEqual(bar_methods, ['f', '_h'])
        self.assertEqual(foo.f(), 'f()')
        self.assertRaises(ValueError, foo.g)
        self.assertEqual(foo._callmethod('f'), 'f()')
        self.assertRaises(RemoteError, foo._callmethod, '_h')
        self.assertEqual(bar.f(), 'f()')
        self.assertEqual(bar._h(), '_h()')
        self.assertEqual(bar._callmethod('f'), 'f()')
        self.assertEqual(bar._callmethod('_h'), '_h()')
        self.assertEqual(list(baz), [i * i for i in range(10)])