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
class HandlerByNameTest(WebTestCase):

    def get_handlers(self):
        return [('/hello1', HelloHandler), ('/hello2', 'tornado.test.web_test.HelloHandler'), url('/hello3', 'tornado.test.web_test.HelloHandler')]

    def test_handler_by_name(self):
        resp = self.fetch('/hello1')
        self.assertEqual(resp.body, b'hello')
        resp = self.fetch('/hello2')
        self.assertEqual(resp.body, b'hello')
        resp = self.fetch('/hello3')
        self.assertEqual(resp.body, b'hello')