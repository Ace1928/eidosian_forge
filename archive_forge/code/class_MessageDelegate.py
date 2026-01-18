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
class MessageDelegate(HTTPMessageDelegate):

    def __init__(self, connection):
        self.connection = connection

    def headers_received(self, start_line, headers):
        content_lengths = {'normal': '10', 'alphabetic': 'foo', 'leading plus': '+10', 'underscore': '1_0'}
        self.connection.write_headers(ResponseStartLine('HTTP/1.1', 200, 'OK'), HTTPHeaders({'Content-Length': content_lengths[headers['x-test']]}))
        self.connection.write(b'1234567890')
        self.connection.finish()