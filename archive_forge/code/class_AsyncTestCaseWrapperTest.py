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
class AsyncTestCaseWrapperTest(unittest.TestCase):

    def test_undecorated_generator(self):

        class Test(AsyncTestCase):

            def test_gen(self):
                yield
        test = Test('test_gen')
        result = unittest.TestResult()
        test.run(result)
        self.assertEqual(len(result.errors), 1)
        self.assertIn('should be decorated', result.errors[0][1])

    @unittest.skipIf(platform.python_implementation() == 'PyPy', 'pypy destructor warnings cannot be silenced')
    @unittest.skipIf(sys.version_info >= (3, 12), 'py312 has its own check for test case returns')
    def test_undecorated_coroutine(self):

        class Test(AsyncTestCase):

            async def test_coro(self):
                pass
        test = Test('test_coro')
        result = unittest.TestResult()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            test.run(result)
        self.assertEqual(len(result.errors), 1)
        self.assertIn('should be decorated', result.errors[0][1])

    def test_undecorated_generator_with_skip(self):

        class Test(AsyncTestCase):

            @unittest.skip("don't run this")
            def test_gen(self):
                yield
        test = Test('test_gen')
        result = unittest.TestResult()
        test.run(result)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.skipped), 1)

    def test_other_return(self):

        class Test(AsyncTestCase):

            def test_other_return(self):
                return 42
        test = Test('test_other_return')
        result = unittest.TestResult()
        test.run(result)
        self.assertEqual(len(result.errors), 1)
        self.assertIn('Return value from test method ignored', result.errors[0][1])

    def test_unwrap(self):

        class Test(AsyncTestCase):

            def test_foo(self):
                pass
        test = Test('test_foo')
        self.assertIs(inspect.unwrap(test.test_foo), test.test_foo.orig_method)