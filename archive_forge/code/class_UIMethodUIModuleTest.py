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
class UIMethodUIModuleTest(SimpleHandlerTestCase):
    """Test that UI methods and modules are created correctly and
    associated with the handler.
    """

    class Handler(RequestHandler):

        def get(self):
            self.render('foo.html')

        def value(self):
            return self.get_argument('value')

    def get_app_kwargs(self):

        def my_ui_method(handler, x):
            return 'In my_ui_method(%s) with handler value %s.' % (x, handler.value())

        class MyModule(UIModule):

            def render(self, x):
                return 'In MyModule(%s) with handler value %s.' % (x, typing.cast(UIMethodUIModuleTest.Handler, self.handler).value())
        loader = DictLoader({'foo.html': '{{ my_ui_method(42) }} {% module MyModule(123) %}'})
        return dict(template_loader=loader, ui_methods={'my_ui_method': my_ui_method}, ui_modules={'MyModule': MyModule})

    def tearDown(self):
        super().tearDown()
        RequestHandler._template_loaders.clear()

    def test_ui_method(self):
        response = self.fetch('/?value=asdf')
        self.assertEqual(response.body, b'In my_ui_method(42) with handler value asdf. In MyModule(123) with handler value asdf.')