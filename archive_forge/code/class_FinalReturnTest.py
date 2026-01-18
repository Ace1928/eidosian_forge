from tornado.concurrent import Future
from tornado import gen
from tornado.escape import (
from tornado.httpclient import HTTPClientError
from tornado.httputil import format_timestamp
from tornado.iostream import IOStream
from tornado import locale
from tornado.locks import Event
from tornado.log import app_log, gen_log
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.template import DictLoader
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, ExpectLog, gen_test
from tornado.test.util import ignore_deprecation
from tornado.util import ObjectDict, unicode_type
from tornado.web import (
import binascii
import contextlib
import copy
import datetime
import email.utils
import gzip
from io import BytesIO
import itertools
import logging
import os
import re
import socket
import typing  # noqa: F401
import unittest
import urllib.parse
class FinalReturnTest(WebTestCase):
    final_return = None

    def get_handlers(self):
        test = self

        class FinishHandler(RequestHandler):

            @gen.coroutine
            def get(self):
                test.final_return = self.finish()
                yield test.final_return

            @gen.coroutine
            def post(self):
                self.write('hello,')
                yield self.flush()
                test.final_return = self.finish('world')
                yield test.final_return

        class RenderHandler(RequestHandler):

            def create_template_loader(self, path):
                return DictLoader({'foo.html': 'hi'})

            @gen.coroutine
            def get(self):
                test.final_return = self.render('foo.html')
        return [('/finish', FinishHandler), ('/render', RenderHandler)]

    def get_app_kwargs(self):
        return dict(template_path='FinalReturnTest')

    def test_finish_method_return_future(self):
        response = self.fetch(self.get_url('/finish'))
        self.assertEqual(response.code, 200)
        self.assertIsInstance(self.final_return, Future)
        self.assertTrue(self.final_return.done())
        response = self.fetch(self.get_url('/finish'), method='POST', body=b'')
        self.assertEqual(response.code, 200)
        self.assertIsInstance(self.final_return, Future)
        self.assertTrue(self.final_return.done())

    def test_render_method_return_future(self):
        response = self.fetch(self.get_url('/render'))
        self.assertEqual(response.code, 200)
        self.assertIsInstance(self.final_return, Future)