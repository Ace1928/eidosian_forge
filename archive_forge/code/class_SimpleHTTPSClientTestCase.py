import collections
from contextlib import closing
import errno
import logging
import os
import re
import socket
import ssl
import sys
import typing  # noqa: F401
from tornado.escape import to_unicode, utf8
from tornado import gen, version
from tornado.httpclient import AsyncHTTPClient
from tornado.httputil import HTTPHeaders, ResponseStartLine
from tornado.ioloop import IOLoop
from tornado.iostream import UnsatisfiableReadError
from tornado.locks import Event
from tornado.log import gen_log
from tornado.netutil import Resolver, bind_sockets
from tornado.simple_httpclient import (
from tornado.test.httpclient_test import (
from tornado.test import httpclient_test
from tornado.testing import (
from tornado.test.util import skipOnTravis, skipIfNoIPv6, refusing_port
from tornado.web import RequestHandler, Application, url, stream_request_body
class SimpleHTTPSClientTestCase(SimpleHTTPClientTestMixin, AsyncHTTPSTestCase):

    def setUp(self):
        super().setUp()
        self.http_client = self.create_client()

    def create_client(self, **kwargs):
        return SimpleAsyncHTTPClient(force_instance=True, defaults=dict(validate_cert=False), **kwargs)

    def test_ssl_options(self):
        resp = self.fetch('/hello', ssl_options={'cert_reqs': ssl.CERT_NONE})
        self.assertEqual(resp.body, b'Hello world!')

    def test_ssl_context(self):
        ssl_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        resp = self.fetch('/hello', ssl_options=ssl_ctx)
        self.assertEqual(resp.body, b'Hello world!')

    def test_ssl_options_handshake_fail(self):
        with ExpectLog(gen_log, 'SSL Error|Uncaught exception', required=False):
            with self.assertRaises(ssl.SSLError):
                self.fetch('/hello', ssl_options=dict(cert_reqs=ssl.CERT_REQUIRED), raise_error=True)

    def test_ssl_context_handshake_fail(self):
        with ExpectLog(gen_log, 'SSL Error|Uncaught exception'):
            ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            with self.assertRaises(ssl.SSLError):
                self.fetch('/hello', ssl_options=ctx, raise_error=True)

    def test_error_logging(self):
        with ExpectLog(gen_log, '.*') as expect_log:
            with self.assertRaises(ssl.SSLError):
                self.fetch('/', validate_cert=True, raise_error=True)
        self.assertFalse(expect_log.logged_stack)