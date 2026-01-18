from tornado.concurrent import Future
from tornado import gen
from tornado.escape import (
from tornado.httpclient import HTTPClientError
from tornado.httputil import format_timestamp
from tornado.iostream import IOStream
from tornado import locale
from tornado.locks import Event
from tornado.log import app_log, gen_log
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.template import DictLoader
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, ExpectLog, gen_test
from tornado.test.util import ignore_deprecation
from tornado.util import ObjectDict, unicode_type
from tornado.web import (
import binascii
import contextlib
import copy
import datetime
import email.utils
import gzip
from io import BytesIO
import itertools
import logging
import os
import re
import socket
import typing  # noqa: F401
import unittest
import urllib.parse
class ConnectionCloseTest(WebTestCase):

    def get_handlers(self):
        self.cleanup_event = Event()
        return [('/', ConnectionCloseHandler, dict(test=self))]

    def test_connection_close(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        s.connect(('127.0.0.1', self.get_http_port()))
        self.stream = IOStream(s)
        self.stream.write(b'GET / HTTP/1.0\r\n\r\n')
        self.wait()
        self.cleanup_event.set()
        self.io_loop.run_sync(lambda: gen.sleep(0))

    def on_handler_waiting(self):
        logging.debug('handler waiting')
        self.stream.close()

    def on_connection_close(self):
        logging.debug('connection closed')
        self.stop()