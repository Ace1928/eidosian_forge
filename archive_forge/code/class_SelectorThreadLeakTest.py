import asyncio
import threading
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from tornado import gen
from tornado.ioloop import IOLoop
from tornado.platform.asyncio import (
from tornado.testing import AsyncTestCase, gen_test
class SelectorThreadLeakTest(unittest.TestCase):

    def setUp(self):
        asyncio.run(self.dummy_tornado_coroutine())
        self.orig_thread_count = threading.active_count()

    def assert_no_thread_leak(self):
        deadline = time.time() + 1
        while time.time() < deadline:
            threads = list(threading.enumerate())
            if len(threads) <= self.orig_thread_count:
                break
            time.sleep(0.1)
        self.assertLessEqual(len(threads), self.orig_thread_count, threads)

    async def dummy_tornado_coroutine(self):
        IOLoop.current()

    def test_asyncio_run(self):
        for i in range(10):
            asyncio.run(self.dummy_tornado_coroutine())
        self.assert_no_thread_leak()

    def test_asyncio_manual(self):
        for i in range(10):
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.dummy_tornado_coroutine())
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
        self.assert_no_thread_leak()

    def test_tornado(self):
        for i in range(10):
            loop = IOLoop(make_current=False)
            loop.run_sync(self.dummy_tornado_coroutine)
            loop.close()
        self.assert_no_thread_leak()