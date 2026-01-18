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
class RedirectHandlerTest(WebTestCase):

    def get_handlers(self):
        return [('/src', WebRedirectHandler, {'url': '/dst'}), ('/src2', WebRedirectHandler, {'url': '/dst2?foo=bar'}), ('/(.*?)/(.*?)/(.*)', WebRedirectHandler, {'url': '/{1}/{0}/{2}'})]

    def test_basic_redirect(self):
        response = self.fetch('/src', follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers['Location'], '/dst')

    def test_redirect_with_argument(self):
        response = self.fetch('/src?foo=bar', follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers['Location'], '/dst?foo=bar')

    def test_redirect_with_appending_argument(self):
        response = self.fetch('/src2?foo2=bar2', follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers['Location'], '/dst2?foo=bar&foo2=bar2')

    def test_redirect_pattern(self):
        response = self.fetch('/a/b/c', follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers['Location'], '/b/a/c')