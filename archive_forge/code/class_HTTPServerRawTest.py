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
class HTTPServerRawTest(AsyncHTTPTestCase):

    def get_app(self):
        return Application([('/echo', EchoHandler)])

    def setUp(self):
        super().setUp()
        self.stream = IOStream(socket.socket())
        self.io_loop.run_sync(lambda: self.stream.connect(('127.0.0.1', self.get_http_port())))

    def tearDown(self):
        self.stream.close()
        super().tearDown()

    def test_empty_request(self):
        self.stream.close()
        self.io_loop.add_timeout(datetime.timedelta(seconds=0.001), self.stop)
        self.wait()

    def test_malformed_first_line_response(self):
        with ExpectLog(gen_log, '.*Malformed HTTP request line', level=logging.INFO):
            self.stream.write(b'asdf\r\n\r\n')
            start_line, headers, response = self.io_loop.run_sync(lambda: read_stream_body(self.stream))
            self.assertEqual('HTTP/1.1', start_line.version)
            self.assertEqual(400, start_line.code)
            self.assertEqual('Bad Request', start_line.reason)

    def test_malformed_first_line_log(self):
        with ExpectLog(gen_log, '.*Malformed HTTP request line', level=logging.INFO):
            self.stream.write(b'asdf\r\n\r\n')
            self.io_loop.add_timeout(datetime.timedelta(seconds=0.05), self.stop)
            self.wait()

    def test_malformed_headers(self):
        with ExpectLog(gen_log, '.*Malformed HTTP message.*no colon in header line', level=logging.INFO):
            self.stream.write(b'GET / HTTP/1.0\r\nasdf\r\n\r\n')
            self.io_loop.add_timeout(datetime.timedelta(seconds=0.05), self.stop)
            self.wait()

    def test_chunked_request_body(self):
        self.stream.write(b'POST /echo HTTP/1.1\nTransfer-Encoding: chunked\nContent-Type: application/x-www-form-urlencoded\n\n4\nfoo=\n3\nbar\n0\n\n'.replace(b'\n', b'\r\n'))
        start_line, headers, response = self.io_loop.run_sync(lambda: read_stream_body(self.stream))
        self.assertEqual(json_decode(response), {'foo': ['bar']})

    def test_chunked_request_uppercase(self):
        self.stream.write(b'POST /echo HTTP/1.1\nTransfer-Encoding: Chunked\nContent-Type: application/x-www-form-urlencoded\n\n4\nfoo=\n3\nbar\n0\n\n'.replace(b'\n', b'\r\n'))
        start_line, headers, response = self.io_loop.run_sync(lambda: read_stream_body(self.stream))
        self.assertEqual(json_decode(response), {'foo': ['bar']})

    def test_chunked_request_body_invalid_size(self):
        self.stream.write(b'POST /echo HTTP/1.1\nTransfer-Encoding: chunked\n\n1_a\n1234567890abcdef1234567890\n0\n\n'.replace(b'\n', b'\r\n'))
        with ExpectLog(gen_log, '.*invalid chunk size', level=logging.INFO):
            start_line, headers, response = self.io_loop.run_sync(lambda: read_stream_body(self.stream))
        self.assertEqual(400, start_line.code)

    @gen_test
    def test_invalid_content_length(self):
        test_cases = [('alphabetic', 'foo'), ('leading plus', '+10'), ('internal underscore', '1_0')]
        for name, value in test_cases:
            with self.subTest(name=name), closing(IOStream(socket.socket())) as stream:
                with ExpectLog(gen_log, '.*Only integer Content-Length is allowed', level=logging.INFO):
                    yield stream.connect(('127.0.0.1', self.get_http_port()))
                    stream.write(utf8(textwrap.dedent(f'                            POST /echo HTTP/1.1\n                            Content-Length: {value}\n                            Connection: close\n\n                            1234567890\n                            ').replace('\n', '\r\n')))
                    yield stream.read_until_close()