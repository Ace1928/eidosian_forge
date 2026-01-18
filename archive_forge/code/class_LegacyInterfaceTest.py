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
class LegacyInterfaceTest(AsyncHTTPTestCase):

    def get_app(self):

        def handle_request(request):
            self.http1 = request.version.startswith('HTTP/1.')
            if not self.http1:
                request.connection.write_headers(ResponseStartLine('', 200, 'OK'), HTTPHeaders())
                request.connection.finish()
                return
            message = b'Hello world'
            request.connection.write(utf8('HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n' % len(message)))
            request.connection.write(message)
            request.connection.finish()
        return handle_request

    def test_legacy_interface(self):
        response = self.fetch('/')
        if not self.http1:
            self.skipTest('requires HTTP/1.x')
        self.assertEqual(response.body, b'Hello world')