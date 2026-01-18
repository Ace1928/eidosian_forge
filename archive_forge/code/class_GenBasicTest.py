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
class GenBasicTest(AsyncTestCase):

    @gen.coroutine
    def delay(self, iterations, arg):
        """Returns arg after a number of IOLoop iterations."""
        for i in range(iterations):
            yield gen.moment
        raise gen.Return(arg)

    @gen.coroutine
    def async_future(self, result):
        yield gen.moment
        return result

    @gen.coroutine
    def async_exception(self, e):
        yield gen.moment
        raise e

    @gen.coroutine
    def add_one_async(self, x):
        yield gen.moment
        raise gen.Return(x + 1)

    def test_no_yield(self):

        @gen.coroutine
        def f():
            pass
        self.io_loop.run_sync(f)

    def test_exception_phase1(self):

        @gen.coroutine
        def f():
            1 / 0
        self.assertRaises(ZeroDivisionError, self.io_loop.run_sync, f)

    def test_exception_phase2(self):

        @gen.coroutine
        def f():
            yield gen.moment
            1 / 0
        self.assertRaises(ZeroDivisionError, self.io_loop.run_sync, f)

    def test_bogus_yield(self):

        @gen.coroutine
        def f():
            yield 42
        self.assertRaises(gen.BadYieldError, self.io_loop.run_sync, f)

    def test_bogus_yield_tuple(self):

        @gen.coroutine
        def f():
            yield (1, 2)
        self.assertRaises(gen.BadYieldError, self.io_loop.run_sync, f)

    def test_reuse(self):

        @gen.coroutine
        def f():
            yield gen.moment
        self.io_loop.run_sync(f)
        self.io_loop.run_sync(f)

    def test_none(self):

        @gen.coroutine
        def f():
            yield None
        self.io_loop.run_sync(f)

    def test_multi(self):

        @gen.coroutine
        def f():
            results = (yield [self.add_one_async(1), self.add_one_async(2)])
            self.assertEqual(results, [2, 3])
        self.io_loop.run_sync(f)

    def test_multi_dict(self):

        @gen.coroutine
        def f():
            results = (yield dict(foo=self.add_one_async(1), bar=self.add_one_async(2)))
            self.assertEqual(results, dict(foo=2, bar=3))
        self.io_loop.run_sync(f)

    def test_multi_delayed(self):

        @gen.coroutine
        def f():
            responses = (yield gen.multi_future([self.delay(3, 'v1'), self.delay(1, 'v2')]))
            self.assertEqual(responses, ['v1', 'v2'])
        self.io_loop.run_sync(f)

    def test_multi_dict_delayed(self):

        @gen.coroutine
        def f():
            responses = (yield gen.multi_future(dict(foo=self.delay(3, 'v1'), bar=self.delay(1, 'v2'))))
            self.assertEqual(responses, dict(foo='v1', bar='v2'))
        self.io_loop.run_sync(f)

    @skipOnTravis
    @gen_test
    def test_multi_performance(self):
        start = time.time()
        yield [gen.moment for i in range(2000)]
        end = time.time()
        self.assertLess(end - start, 1.0)

    @gen_test
    def test_multi_empty(self):
        x = (yield [])
        self.assertTrue(isinstance(x, list))
        y = (yield {})
        self.assertTrue(isinstance(y, dict))

    @gen_test
    def test_future(self):
        result = (yield self.async_future(1))
        self.assertEqual(result, 1)

    @gen_test
    def test_multi_future(self):
        results = (yield [self.async_future(1), self.async_future(2)])
        self.assertEqual(results, [1, 2])

    @gen_test
    def test_multi_future_duplicate(self):
        f = self.async_future(2)
        results = (yield [self.async_future(1), f, self.async_future(3), f])
        self.assertEqual(results, [1, 2, 3, 2])

    @gen_test
    def test_multi_dict_future(self):
        results = (yield dict(foo=self.async_future(1), bar=self.async_future(2)))
        self.assertEqual(results, dict(foo=1, bar=2))

    @gen_test
    def test_multi_exceptions(self):
        with ExpectLog(app_log, 'Multiple exceptions in yield list'):
            with self.assertRaises(RuntimeError) as cm:
                yield gen.Multi([self.async_exception(RuntimeError('error 1')), self.async_exception(RuntimeError('error 2'))])
        self.assertEqual(str(cm.exception), 'error 1')
        with self.assertRaises(RuntimeError):
            yield gen.Multi([self.async_exception(RuntimeError('error 1')), self.async_future(2)])
        with self.assertRaises(RuntimeError):
            yield gen.Multi([self.async_exception(RuntimeError('error 1')), self.async_exception(RuntimeError('error 2'))], quiet_exceptions=RuntimeError)

    @gen_test
    def test_multi_future_exceptions(self):
        with ExpectLog(app_log, 'Multiple exceptions in yield list'):
            with self.assertRaises(RuntimeError) as cm:
                yield [self.async_exception(RuntimeError('error 1')), self.async_exception(RuntimeError('error 2'))]
        self.assertEqual(str(cm.exception), 'error 1')
        with self.assertRaises(RuntimeError):
            yield [self.async_exception(RuntimeError('error 1')), self.async_future(2)]
        with self.assertRaises(RuntimeError):
            yield gen.multi_future([self.async_exception(RuntimeError('error 1')), self.async_exception(RuntimeError('error 2'))], quiet_exceptions=RuntimeError)

    def test_sync_raise_return(self):

        @gen.coroutine
        def f():
            raise gen.Return()
        self.io_loop.run_sync(f)

    def test_async_raise_return(self):

        @gen.coroutine
        def f():
            yield gen.moment
            raise gen.Return()
        self.io_loop.run_sync(f)

    def test_sync_raise_return_value(self):

        @gen.coroutine
        def f():
            raise gen.Return(42)
        self.assertEqual(42, self.io_loop.run_sync(f))

    def test_sync_raise_return_value_tuple(self):

        @gen.coroutine
        def f():
            raise gen.Return((1, 2))
        self.assertEqual((1, 2), self.io_loop.run_sync(f))

    def test_async_raise_return_value(self):

        @gen.coroutine
        def f():
            yield gen.moment
            raise gen.Return(42)
        self.assertEqual(42, self.io_loop.run_sync(f))

    def test_async_raise_return_value_tuple(self):

        @gen.coroutine
        def f():
            yield gen.moment
            raise gen.Return((1, 2))
        self.assertEqual((1, 2), self.io_loop.run_sync(f))