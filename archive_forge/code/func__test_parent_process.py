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
def _test_parent_process(self):
    if self.TYPE == 'threads':
        self.skipTest('test not appropriate for {}'.format(self.TYPE))
    rconn, wconn = self.Pipe(duplex=False)
    p = self.Process(target=self._test_create_grandchild_process, args=(wconn,))
    p.start()
    if not rconn.poll(timeout=support.LONG_TIMEOUT):
        raise AssertionError('Could not communicate with child process')
    parent_process_status = rconn.recv()
    self.assertEqual(parent_process_status, 'alive')
    p.terminate()
    p.join()
    if not rconn.poll(timeout=support.LONG_TIMEOUT):
        raise AssertionError('Could not communicate with child process')
    parent_process_status = rconn.recv()
    self.assertEqual(parent_process_status, 'not alive')