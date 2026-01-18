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
class GzipTest(GzipBaseTest, AsyncHTTPTestCase):

    def get_httpserver_options(self):
        return dict(decompress_request=True)

    def test_gzip(self):
        response = self.post_gzip('foo=bar')
        self.assertEqual(json_decode(response.body), {'foo': ['bar']})

    def test_gzip_case_insensitive(self):
        bytesio = BytesIO()
        gzip_file = gzip.GzipFile(mode='w', fileobj=bytesio)
        gzip_file.write(utf8('foo=bar'))
        gzip_file.close()
        compressed_body = bytesio.getvalue()
        response = self.fetch('/', method='POST', body=compressed_body, headers={'Content-Encoding': 'GZIP'})
        self.assertEqual(json_decode(response.body), {'foo': ['bar']})