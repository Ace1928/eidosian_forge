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
class WaitIteratorTest(AsyncTestCase):

    @gen_test
    def test_empty_iterator(self):
        g = gen.WaitIterator()
        self.assertTrue(g.done(), 'empty generator iterated')
        with self.assertRaises(ValueError):
            g = gen.WaitIterator(Future(), bar=Future())
        self.assertEqual(g.current_index, None, 'bad nil current index')
        self.assertEqual(g.current_future, None, 'bad nil current future')

    @gen_test
    def test_already_done(self):
        f1 = Future()
        f2 = Future()
        f3 = Future()
        f1.set_result(24)
        f2.set_result(42)
        f3.set_result(84)
        g = gen.WaitIterator(f1, f2, f3)
        i = 0
        while not g.done():
            r = (yield g.next())
            if i == 0:
                self.assertEqual(g.current_index, 0)
                self.assertIs(g.current_future, f1)
                self.assertEqual(r, 24)
            elif i == 1:
                self.assertEqual(g.current_index, 1)
                self.assertIs(g.current_future, f2)
                self.assertEqual(r, 42)
            elif i == 2:
                self.assertEqual(g.current_index, 2)
                self.assertIs(g.current_future, f3)
                self.assertEqual(r, 84)
            i += 1
        self.assertEqual(g.current_index, None, 'bad nil current index')
        self.assertEqual(g.current_future, None, 'bad nil current future')
        dg = gen.WaitIterator(f1=f1, f2=f2)
        while not dg.done():
            dr = (yield dg.next())
            if dg.current_index == 'f1':
                self.assertTrue(dg.current_future == f1 and dr == 24, 'WaitIterator dict status incorrect')
            elif dg.current_index == 'f2':
                self.assertTrue(dg.current_future == f2 and dr == 42, 'WaitIterator dict status incorrect')
            else:
                self.fail('got bad WaitIterator index {}'.format(dg.current_index))
            i += 1
        self.assertEqual(dg.current_index, None, 'bad nil current index')
        self.assertEqual(dg.current_future, None, 'bad nil current future')

    def finish_coroutines(self, iteration, futures):
        if iteration == 3:
            futures[2].set_result(24)
        elif iteration == 5:
            futures[0].set_exception(ZeroDivisionError())
        elif iteration == 8:
            futures[1].set_result(42)
            futures[3].set_result(84)
        if iteration < 8:
            self.io_loop.add_callback(self.finish_coroutines, iteration + 1, futures)

    @gen_test
    def test_iterator(self):
        futures = [Future(), Future(), Future(), Future()]
        self.finish_coroutines(0, futures)
        g = gen.WaitIterator(*futures)
        i = 0
        while not g.done():
            try:
                r = (yield g.next())
            except ZeroDivisionError:
                self.assertIs(g.current_future, futures[0], 'exception future invalid')
            else:
                if i == 0:
                    self.assertEqual(r, 24, 'iterator value incorrect')
                    self.assertEqual(g.current_index, 2, 'wrong index')
                elif i == 2:
                    self.assertEqual(r, 42, 'iterator value incorrect')
                    self.assertEqual(g.current_index, 1, 'wrong index')
                elif i == 3:
                    self.assertEqual(r, 84, 'iterator value incorrect')
                    self.assertEqual(g.current_index, 3, 'wrong index')
            i += 1

    @gen_test
    def test_iterator_async_await(self):
        futures = [Future(), Future(), Future(), Future()]
        self.finish_coroutines(0, futures)
        self.finished = False

        async def f():
            i = 0
            g = gen.WaitIterator(*futures)
            try:
                async for r in g:
                    if i == 0:
                        self.assertEqual(r, 24, 'iterator value incorrect')
                        self.assertEqual(g.current_index, 2, 'wrong index')
                    else:
                        raise Exception('expected exception on iteration 1')
                    i += 1
            except ZeroDivisionError:
                i += 1
            async for r in g:
                if i == 2:
                    self.assertEqual(r, 42, 'iterator value incorrect')
                    self.assertEqual(g.current_index, 1, 'wrong index')
                elif i == 3:
                    self.assertEqual(r, 84, 'iterator value incorrect')
                    self.assertEqual(g.current_index, 3, 'wrong index')
                else:
                    raise Exception("didn't expect iteration %d" % i)
                i += 1
            self.finished = True
        yield f()
        self.assertTrue(self.finished)

    @gen_test
    def test_no_ref(self):
        yield gen.with_timeout(datetime.timedelta(seconds=0.1), gen.WaitIterator(gen.sleep(0)).next())