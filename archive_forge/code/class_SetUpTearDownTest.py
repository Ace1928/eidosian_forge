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
class SetUpTearDownTest(unittest.TestCase):

    def test_set_up_tear_down(self):
        """
        This test makes sure that AsyncTestCase calls super methods for
        setUp and tearDown.

        InheritBoth is a subclass of both AsyncTestCase and
        SetUpTearDown, with the ordering so that the super of
        AsyncTestCase will be SetUpTearDown.
        """
        events = []
        result = unittest.TestResult()

        class SetUpTearDown(unittest.TestCase):

            def setUp(self):
                events.append('setUp')

            def tearDown(self):
                events.append('tearDown')

        class InheritBoth(AsyncTestCase, SetUpTearDown):

            def test(self):
                events.append('test')
        InheritBoth('test').run(result)
        expected = ['setUp', 'test', 'tearDown']
        self.assertEqual(expected, events)