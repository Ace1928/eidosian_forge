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
class BodyLimitsTest(AsyncHTTPTestCase):

    def get_app(self):

        class BufferedHandler(RequestHandler):

            def put(self):
                self.write(str(len(self.request.body)))

        @stream_request_body
        class StreamingHandler(RequestHandler):

            def initialize(self):
                self.bytes_read = 0

            def prepare(self):
                conn = typing.cast(HTTP1Connection, self.request.connection)
                if 'expected_size' in self.request.arguments:
                    conn.set_max_body_size(int(self.get_argument('expected_size')))
                if 'body_timeout' in self.request.arguments:
                    conn.set_body_timeout(float(self.get_argument('body_timeout')))

            def data_received(self, data):
                self.bytes_read += len(data)

            def put(self):
                self.write(str(self.bytes_read))
        return Application([('/buffered', BufferedHandler), ('/streaming', StreamingHandler)])

    def get_httpserver_options(self):
        return dict(body_timeout=3600, max_body_size=4096)

    def get_http_client(self):
        return SimpleAsyncHTTPClient()

    def test_small_body(self):
        response = self.fetch('/buffered', method='PUT', body=b'a' * 4096)
        self.assertEqual(response.body, b'4096')
        response = self.fetch('/streaming', method='PUT', body=b'a' * 4096)
        self.assertEqual(response.body, b'4096')

    def test_large_body_buffered(self):
        with ExpectLog(gen_log, '.*Content-Length too long', level=logging.INFO):
            response = self.fetch('/buffered', method='PUT', body=b'a' * 10240)
        self.assertEqual(response.code, 400)

    @unittest.skipIf(os.name == 'nt', 'flaky on windows')
    def test_large_body_buffered_chunked(self):
        with ExpectLog(gen_log, '.*chunked body too large', level=logging.INFO):
            response = self.fetch('/buffered', method='PUT', body_producer=lambda write: write(b'a' * 10240))
        self.assertEqual(response.code, 400)

    def test_large_body_streaming(self):
        with ExpectLog(gen_log, '.*Content-Length too long', level=logging.INFO):
            response = self.fetch('/streaming', method='PUT', body=b'a' * 10240)
        self.assertEqual(response.code, 400)

    @unittest.skipIf(os.name == 'nt', 'flaky on windows')
    def test_large_body_streaming_chunked(self):
        with ExpectLog(gen_log, '.*chunked body too large', level=logging.INFO):
            response = self.fetch('/streaming', method='PUT', body_producer=lambda write: write(b'a' * 10240))
        self.assertEqual(response.code, 400)

    def test_large_body_streaming_override(self):
        response = self.fetch('/streaming?expected_size=10240', method='PUT', body=b'a' * 10240)
        self.assertEqual(response.body, b'10240')

    def test_large_body_streaming_chunked_override(self):
        response = self.fetch('/streaming?expected_size=10240', method='PUT', body_producer=lambda write: write(b'a' * 10240))
        self.assertEqual(response.body, b'10240')

    @gen_test
    def test_timeout(self):
        stream = IOStream(socket.socket())
        try:
            yield stream.connect(('127.0.0.1', self.get_http_port()))
            stream.write(b'PUT /streaming?body_timeout=0.1 HTTP/1.0\r\nContent-Length: 42\r\n\r\n')
            with ExpectLog(gen_log, 'Timeout reading body', level=logging.INFO):
                response = (yield stream.read_until_close())
            self.assertEqual(response, b'')
        finally:
            stream.close()

    @gen_test
    def test_body_size_override_reset(self):
        stream = IOStream(socket.socket())
        try:
            yield stream.connect(('127.0.0.1', self.get_http_port()))
            stream.write(b'PUT /streaming?expected_size=10240 HTTP/1.1\r\nContent-Length: 10240\r\n\r\n')
            stream.write(b'a' * 10240)
            start_line, headers, response = (yield read_stream_body(stream))
            self.assertEqual(response, b'10240')
            stream.write(b'PUT /streaming HTTP/1.1\r\nContent-Length: 10240\r\n\r\n')
            with ExpectLog(gen_log, '.*Content-Length too long', level=logging.INFO):
                data = (yield stream.read_until_close())
            self.assertEqual(data, b'HTTP/1.1 400 Bad Request\r\n\r\n')
        finally:
            stream.close()