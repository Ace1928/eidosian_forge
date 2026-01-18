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
class GenWebTest(AsyncHTTPTestCase):

    def get_app(self):
        return Application([('/coroutine_sequence', GenCoroutineSequenceHandler), ('/coroutine_unfinished_sequence', GenCoroutineUnfinishedSequenceHandler), ('/undecorated_coroutine', UndecoratedCoroutinesHandler), ('/async_prepare_error', AsyncPrepareErrorHandler), ('/native_coroutine', NativeCoroutineHandler)])

    def test_coroutine_sequence_handler(self):
        response = self.fetch('/coroutine_sequence')
        self.assertEqual(response.body, b'123')

    def test_coroutine_unfinished_sequence_handler(self):
        response = self.fetch('/coroutine_unfinished_sequence')
        self.assertEqual(response.body, b'123')

    def test_undecorated_coroutines(self):
        response = self.fetch('/undecorated_coroutine')
        self.assertEqual(response.body, b'123')

    def test_async_prepare_error_handler(self):
        response = self.fetch('/async_prepare_error')
        self.assertEqual(response.code, 403)

    def test_native_coroutine_handler(self):
        response = self.fetch('/native_coroutine')
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body, b'ok')