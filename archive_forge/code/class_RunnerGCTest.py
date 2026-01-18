import asyncio
from concurrent import futures
import gc
import datetime
import platform
import sys
import time
import weakref
import unittest
from tornado.concurrent import Future
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, ExpectLog, gen_test
from tornado.test.util import skipOnTravis, skipNotCPython
from tornado.web import Application, RequestHandler, HTTPError
from tornado import gen
import typing
class RunnerGCTest(AsyncTestCase):

    def is_pypy3(self):
        return platform.python_implementation() == 'PyPy' and sys.version_info > (3,)

    @gen_test
    def test_gc(self):
        weakref_scope = [None]

        def callback():
            gc.collect(2)
            weakref_scope[0]().set_result(123)

        @gen.coroutine
        def tester():
            fut = Future()
            weakref_scope[0] = weakref.ref(fut)
            self.io_loop.add_callback(callback)
            yield fut
        yield gen.with_timeout(datetime.timedelta(seconds=0.2), tester())

    def test_gc_infinite_coro(self):
        loop = self.get_new_ioloop()
        result = []
        wfut = []

        @gen.coroutine
        def infinite_coro():
            try:
                while True:
                    yield gen.sleep(0.001)
                    result.append(True)
            finally:
                result.append(None)

        @gen.coroutine
        def do_something():
            fut = infinite_coro()
            fut._refcycle = fut
            wfut.append(weakref.ref(fut))
            yield gen.sleep(0.2)
        loop.run_sync(do_something)
        loop.close()
        gc.collect()
        self.assertIs(wfut[0](), None)
        self.assertGreaterEqual(len(result), 2)
        if not self.is_pypy3():
            self.assertIs(result[-1], None)

    def test_gc_infinite_async_await(self):
        import asyncio

        async def infinite_coro(result):
            try:
                while True:
                    await gen.sleep(0.001)
                    result.append(True)
            finally:
                result.append(None)
        loop = self.get_new_ioloop()
        result = []
        wfut = []

        @gen.coroutine
        def do_something():
            fut = asyncio.get_event_loop().create_task(infinite_coro(result))
            fut._refcycle = fut
            wfut.append(weakref.ref(fut))
            yield gen.sleep(0.2)
        loop.run_sync(do_something)
        with ExpectLog('asyncio', 'Task was destroyed but it is pending'):
            loop.close()
            gc.collect()
        self.assertIs(wfut[0](), None)
        self.assertGreaterEqual(len(result), 2)
        if not self.is_pypy3():
            self.assertIs(result[-1], None)

    def test_multi_moment(self):

        @gen.coroutine
        def wait_a_moment():
            result = (yield gen.multi([gen.moment, gen.moment]))
            raise gen.Return(result)
        loop = self.get_new_ioloop()
        result = loop.run_sync(wait_a_moment)
        self.assertEqual(result, [None, None])