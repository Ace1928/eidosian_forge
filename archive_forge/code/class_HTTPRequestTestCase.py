import base64
import binascii
from contextlib import closing
import copy
import gzip
import threading
import datetime
from io import BytesIO
import subprocess
import sys
import time
import typing  # noqa: F401
import unicodedata
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado import gen
from tornado.httpclient import (
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.iostream import IOStream
from tornado.log import gen_log, app_log
from tornado import netutil
from tornado.testing import AsyncHTTPTestCase, bind_unused_port, gen_test, ExpectLog
from tornado.test.util import skipOnTravis, ignore_deprecation
from tornado.web import Application, RequestHandler, url
from tornado.httputil import format_timestamp, HTTPHeaders
class HTTPRequestTestCase(unittest.TestCase):

    def test_headers(self):
        request = HTTPRequest('http://example.com', headers={'foo': 'bar'})
        self.assertEqual(request.headers, {'foo': 'bar'})

    def test_headers_setter(self):
        request = HTTPRequest('http://example.com')
        request.headers = {'bar': 'baz'}
        self.assertEqual(request.headers, {'bar': 'baz'})

    def test_null_headers_setter(self):
        request = HTTPRequest('http://example.com')
        request.headers = None
        self.assertEqual(request.headers, {})

    def test_body(self):
        request = HTTPRequest('http://example.com', body='foo')
        self.assertEqual(request.body, utf8('foo'))

    def test_body_setter(self):
        request = HTTPRequest('http://example.com')
        request.body = 'foo'
        self.assertEqual(request.body, utf8('foo'))

    def test_if_modified_since(self):
        http_date = datetime.datetime.now(datetime.timezone.utc)
        request = HTTPRequest('http://example.com', if_modified_since=http_date)
        self.assertEqual(request.headers, {'If-Modified-Since': format_timestamp(http_date)})

    def test_if_modified_since_naive_deprecated(self):
        with ignore_deprecation():
            http_date = datetime.datetime.utcnow()
        request = HTTPRequest('http://example.com', if_modified_since=http_date)
        self.assertEqual(request.headers, {'If-Modified-Since': format_timestamp(http_date)})