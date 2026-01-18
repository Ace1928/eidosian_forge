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
class SyncHTTPClientTest(unittest.TestCase):

    def setUp(self):
        self.server_ioloop = IOLoop(make_current=False)
        event = threading.Event()

        @gen.coroutine
        def init_server():
            sock, self.port = bind_unused_port()
            app = Application([('/', HelloWorldHandler)])
            self.server = HTTPServer(app)
            self.server.add_socket(sock)
            event.set()

        def start():
            self.server_ioloop.run_sync(init_server)
            self.server_ioloop.start()
        self.server_thread = threading.Thread(target=start)
        self.server_thread.start()
        event.wait()
        self.http_client = HTTPClient()

    def tearDown(self):

        def stop_server():
            self.server.stop()

            @gen.coroutine
            def slow_stop():
                yield self.server.close_all_connections()
                for i in range(5):
                    yield
                self.server_ioloop.stop()
            self.server_ioloop.add_callback(slow_stop)
        self.server_ioloop.add_callback(stop_server)
        self.server_thread.join()
        self.http_client.close()
        self.server_ioloop.close(all_fds=True)

    def get_url(self, path):
        return 'http://127.0.0.1:%d%s' % (self.port, path)

    def test_sync_client(self):
        response = self.http_client.fetch(self.get_url('/'))
        self.assertEqual(b'Hello world!', response.body)

    def test_sync_client_error(self):
        with self.assertRaises(HTTPError) as assertion:
            self.http_client.fetch(self.get_url('/notfound'))
        self.assertEqual(assertion.exception.code, 404)