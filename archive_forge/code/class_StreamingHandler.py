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