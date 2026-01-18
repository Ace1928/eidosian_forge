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
import functools
import signal
import array
import socket
import random
import logging
import subprocess
import struct
import operator
import pathlib
import pickle #XXX: use dill?
import weakref
import warnings
import test.support
import test.support.script_helper
from test import support
from test.support import hashlib_helper
from test.support import import_helper
from test.support import os_helper
from test.support import script_helper
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
from multiprocess.connection import wait, AuthenticationError
from multiprocess import util
from multiprocess.managers import BaseManager, BaseProxy, RemoteError
class TestInternalDecorators(unittest.TestCase):
    """Logic within a test suite that could errantly skip tests? Test it!"""

    @unittest.skipIf(sys.platform == 'win32', 'test requires that fork exists.')
    def test_only_run_in_spawn_testsuite(self):
        if multiprocessing.get_start_method() != 'spawn':
            raise unittest.SkipTest('only run in test_multiprocessing_spawn.')
        try:

            @only_run_in_spawn_testsuite('testing this decorator')
            def return_four_if_spawn():
                return 4
        except Exception as err:
            self.fail(f'expected decorated `def` not to raise; caught {err}')
        orig_start_method = multiprocessing.get_start_method(allow_none=True)
        try:
            multiprocessing.set_start_method('spawn', force=True)
            self.assertEqual(return_four_if_spawn(), 4)
            multiprocessing.set_start_method('fork', force=True)
            with self.assertRaises(unittest.SkipTest) as ctx:
                return_four_if_spawn()
            self.assertIn('testing this decorator', str(ctx.exception))
            self.assertIn('start_method=', str(ctx.exception))
        finally:
            multiprocessing.set_start_method(orig_start_method, force=True)