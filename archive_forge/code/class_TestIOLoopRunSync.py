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
class TestIOLoopRunSync(unittest.TestCase):

    def setUp(self):
        self.io_loop = IOLoop(make_current=False)

    def tearDown(self):
        self.io_loop.close()

    def test_sync_result(self):
        with self.assertRaises(gen.BadYieldError):
            self.io_loop.run_sync(lambda: 42)

    def test_sync_exception(self):
        with self.assertRaises(ZeroDivisionError):
            self.io_loop.run_sync(lambda: 1 / 0)

    def test_async_result(self):

        @gen.coroutine
        def f():
            yield gen.moment
            raise gen.Return(42)
        self.assertEqual(self.io_loop.run_sync(f), 42)

    def test_async_exception(self):

        @gen.coroutine
        def f():
            yield gen.moment
            1 / 0
        with self.assertRaises(ZeroDivisionError):
            self.io_loop.run_sync(f)

    def test_current(self):

        def f():
            self.assertIs(IOLoop.current(), self.io_loop)
        self.io_loop.run_sync(f)

    def test_timeout(self):

        @gen.coroutine
        def f():
            yield gen.sleep(1)
        self.assertRaises(TimeoutError, self.io_loop.run_sync, f, timeout=0.01)

    def test_native_coroutine(self):

        @gen.coroutine
        def f1():
            yield gen.moment

        async def f2():
            await f1()
        self.io_loop.run_sync(f2)