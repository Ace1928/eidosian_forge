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
class StaticFileWithPathTest(WebTestCase):

    def get_app_kwargs(self):
        return dict(static_path=relpath('static'), static_handler_args=dict(default_filename='index.html'))

    def get_handlers(self):
        return [('/foo/(.*)', StaticFileHandler, {'path': relpath('templates/')})]

    def test_serve(self):
        response = self.fetch('/foo/utf8.html')
        self.assertEqual(response.body, b'H\xc3\xa9llo\n')