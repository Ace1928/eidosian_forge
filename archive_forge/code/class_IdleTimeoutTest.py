from tornado import gen, netutil
from tornado.escape import (
from tornado.http1connection import HTTP1Connection
from tornado.httpclient import HTTPError
from tornado.httpserver import HTTPServer
from tornado.httputil import (
from tornado.iostream import IOStream
from tornado.locks import Event
from tornado.log import gen_log, app_log
from tornado.netutil import ssl_options_to_context
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.testing import (
from tornado.test.util import skipOnTravis
from tornado.web import Application, RequestHandler, stream_request_body
from contextlib import closing
import datetime
import gzip
import logging
import os
import shutil
import socket
import ssl
import sys
import tempfile
import textwrap
import unittest
import urllib.parse
from io import BytesIO
import typing
@skipOnTravis
class IdleTimeoutTest(AsyncHTTPTestCase):

    def get_app(self):
        return Application([('/', HelloWorldRequestHandler)])

    def get_httpserver_options(self):
        return dict(idle_connection_timeout=0.1)

    def setUp(self):
        super().setUp()
        self.streams = []

    def tearDown(self):
        super().tearDown()
        for stream in self.streams:
            stream.close()

    @gen.coroutine
    def connect(self):
        stream = IOStream(socket.socket())
        yield stream.connect(('127.0.0.1', self.get_http_port()))
        self.streams.append(stream)
        raise gen.Return(stream)

    @gen_test
    def test_unused_connection(self):
        stream = (yield self.connect())
        event = Event()
        stream.set_close_callback(event.set)
        yield event.wait()

    @gen_test
    def test_idle_after_use(self):
        stream = (yield self.connect())
        event = Event()
        stream.set_close_callback(event.set)
        for i in range(2):
            stream.write(b'GET / HTTP/1.1\r\n\r\n')
            yield stream.read_until(b'\r\n\r\n')
            data = (yield stream.read_bytes(11))
            self.assertEqual(data, b'Hello world')
        yield event.wait()