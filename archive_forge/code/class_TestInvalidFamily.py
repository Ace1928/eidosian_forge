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
class TestInvalidFamily(unittest.TestCase):

    @unittest.skipIf(WIN32, 'skipped on Windows')
    def test_invalid_family(self):
        with self.assertRaises(ValueError):
            multiprocessing.connection.Listener('\\\\.\\test')

    @unittest.skipUnless(WIN32, 'skipped on non-Windows platforms')
    def test_invalid_family_win32(self):
        with self.assertRaises(ValueError):
            multiprocessing.connection.Listener('/var/test.pipe')