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
@unittest.skipIf(not hasattr(socket, 'AF_UNIX') or sys.platform == 'cygwin', 'unix sockets not supported on this platform')
class UnixSocketTest(AsyncTestCase):
    """HTTPServers can listen on Unix sockets too.

    Why would you want to do this?  Nginx can proxy to backends listening
    on unix sockets, for one thing (and managing a namespace for unix
    sockets can be easier than managing a bunch of TCP port numbers).

    Unfortunately, there's no way to specify a unix socket in a url for
    an HTTP client, so we have to test this by hand.
    """

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()
        self.sockfile = os.path.join(self.tmpdir, 'test.sock')
        sock = netutil.bind_unix_socket(self.sockfile)
        app = Application([('/hello', HelloWorldRequestHandler)])
        self.server = HTTPServer(app)
        self.server.add_socket(sock)
        self.stream = IOStream(socket.socket(socket.AF_UNIX))
        self.io_loop.run_sync(lambda: self.stream.connect(self.sockfile))

    def tearDown(self):
        self.stream.close()
        self.io_loop.run_sync(self.server.close_all_connections)
        self.server.stop()
        shutil.rmtree(self.tmpdir)
        super().tearDown()

    @gen_test
    def test_unix_socket(self):
        self.stream.write(b'GET /hello HTTP/1.0\r\n\r\n')
        response = (yield self.stream.read_until(b'\r\n'))
        self.assertEqual(response, b'HTTP/1.1 200 OK\r\n')
        header_data = (yield self.stream.read_until(b'\r\n\r\n'))
        headers = HTTPHeaders.parse(header_data.decode('latin1'))
        body = (yield self.stream.read_bytes(int(headers['Content-Length'])))
        self.assertEqual(body, b'Hello world')

    @gen_test
    def test_unix_socket_bad_request(self):
        with ExpectLog(gen_log, 'Malformed HTTP message from', level=logging.INFO):
            self.stream.write(b'garbage\r\n\r\n')
            response = (yield self.stream.read_until_close())
        self.assertEqual(response, b'HTTP/1.1 400 Bad Request\r\n\r\n')