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
@hashlib_helper.requires_hashdigest('md5')
@hashlib_helper.requires_hashdigest('sha256')
class ChallengeResponseTest(unittest.TestCase):
    authkey = b'supadupasecretkey'

    def create_response(self, message):
        return multiprocessing.connection._create_response(self.authkey, message)

    def verify_challenge(self, message, response):
        return multiprocessing.connection._verify_challenge(self.authkey, message, response)

    def test_challengeresponse(self):
        for algo in [None, 'md5', 'sha256']:
            with self.subTest(f'algo={algo!r}'):
                msg = b'is-twenty-bytes-long'
                if algo:
                    prefix = b'{%s}' % algo.encode('ascii')
                else:
                    prefix = b''
                msg = prefix + msg
                response = self.create_response(msg)
                if not response.startswith(prefix):
                    self.fail(response)
                self.verify_challenge(msg, response)