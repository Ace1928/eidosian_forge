import asyncio
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
from collections.abc import Generator
import contextlib
import datetime
import functools
import socket
import subprocess
import sys
import threading
import time
import types
from unittest import mock
import unittest
from tornado.escape import native_str
from tornado import gen
from tornado.ioloop import IOLoop, TimeoutError, PeriodicCallback
from tornado.log import app_log
from tornado.testing import (
from tornado.test.util import (
from tornado.concurrent import Future
import typing
class TestIOLoopCurrent(unittest.TestCase):

    def setUp(self):
        setup_with_context_manager(self, ignore_deprecation())
        self.io_loop = None
        IOLoop.clear_current()

    def tearDown(self):
        if self.io_loop is not None:
            self.io_loop.close()

    def test_non_current(self):
        self.io_loop = IOLoop(make_current=False)
        self.assertIsNone(IOLoop.current(instance=False))
        for i in range(3):

            def f():
                self.current_io_loop = IOLoop.current()
                assert self.io_loop is not None
                self.io_loop.stop()
            self.io_loop.add_callback(f)
            self.io_loop.start()
            self.assertIs(self.current_io_loop, self.io_loop)
            self.assertIsNone(IOLoop.current(instance=False))

    def test_force_current(self):
        self.io_loop = IOLoop(make_current=True)
        self.assertIs(self.io_loop, IOLoop.current())