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
class RaiseWithReasonTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            raise HTTPError(682, reason='Foo')

    def get_http_client(self):
        return SimpleAsyncHTTPClient()

    def test_raise_with_reason(self):
        response = self.fetch('/')
        self.assertEqual(response.code, 682)
        self.assertEqual(response.reason, 'Foo')
        self.assertIn(b'682: Foo', response.body)

    def test_httperror_str(self):
        self.assertEqual(str(HTTPError(682, reason='Foo')), 'HTTP 682: Foo')

    def test_httperror_str_from_httputil(self):
        self.assertEqual(str(HTTPError(682)), 'HTTP 682: Unknown')