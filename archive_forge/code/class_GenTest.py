from tornado import gen, ioloop
from tornado.httpserver import HTTPServer
from tornado.locks import Event
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, bind_unused_port, gen_test
from tornado.web import Application
import asyncio
import contextlib
import inspect
import gc
import os
import platform
import sys
import traceback
import unittest
import warnings
class GenTest(AsyncTestCase):

    def setUp(self):
        super().setUp()
        self.finished = False

    def tearDown(self):
        self.assertTrue(self.finished)
        super().tearDown()

    @gen_test
    def test_sync(self):
        self.finished = True

    @gen_test
    def test_async(self):
        yield gen.moment
        self.finished = True

    def test_timeout(self):

        @gen_test(timeout=0.1)
        def test(self):
            yield gen.sleep(1)
        try:
            test(self)
            self.fail('did not get expected exception')
        except ioloop.TimeoutError:
            self.assertIn('gen.sleep(1)', traceback.format_exc())
        self.finished = True

    def test_no_timeout(self):

        @gen_test(timeout=1)
        def test(self):
            yield gen.sleep(0.1)
        test(self)
        self.finished = True

    def test_timeout_environment_variable(self):

        @gen_test(timeout=0.5)
        def test_long_timeout(self):
            yield gen.sleep(0.25)
        with set_environ('ASYNC_TEST_TIMEOUT', '0.1'):
            test_long_timeout(self)
        self.finished = True

    def test_no_timeout_environment_variable(self):

        @gen_test(timeout=0.01)
        def test_short_timeout(self):
            yield gen.sleep(1)
        with set_environ('ASYNC_TEST_TIMEOUT', '0.1'):
            with self.assertRaises(ioloop.TimeoutError):
                test_short_timeout(self)
        self.finished = True

    def test_with_method_args(self):

        @gen_test
        def test_with_args(self, *args):
            self.assertEqual(args, ('test',))
            yield gen.moment
        test_with_args(self, 'test')
        self.finished = True

    def test_with_method_kwargs(self):

        @gen_test
        def test_with_kwargs(self, **kwargs):
            self.assertDictEqual(kwargs, {'test': 'test'})
            yield gen.moment
        test_with_kwargs(self, test='test')
        self.finished = True

    def test_native_coroutine(self):

        @gen_test
        async def test(self):
            self.finished = True
        test(self)

    def test_native_coroutine_timeout(self):

        @gen_test(timeout=0.1)
        async def test(self):
            await gen.sleep(1)
        try:
            test(self)
            self.fail('did not get expected exception')
        except ioloop.TimeoutError:
            self.finished = True