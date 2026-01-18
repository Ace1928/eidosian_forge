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
class AcceptLanguageTest(WebTestCase):
    """Test evaluation of Accept-Language header"""

    def get_handlers(self):
        locale.load_gettext_translations(os.path.join(os.path.dirname(__file__), 'gettext_translations'), 'tornado_test')

        class AcceptLanguageHandler(RequestHandler):

            def get(self):
                self.set_header('Content-Language', self.get_browser_locale().code.replace('_', '-'))
                self.finish(b'')
        return [('/', AcceptLanguageHandler)]

    def test_accept_language(self):
        response = self.fetch('/', headers={'Accept-Language': 'fr-FR;q=0.9'})
        self.assertEqual(response.headers['Content-Language'], 'fr-FR')
        response = self.fetch('/', headers={'Accept-Language': 'fr-FR; q=0.9'})
        self.assertEqual(response.headers['Content-Language'], 'fr-FR')

    def test_accept_language_ignore(self):
        response = self.fetch('/', headers={'Accept-Language': 'fr-FR;q=0'})
        self.assertEqual(response.headers['Content-Language'], 'en-US')

    def test_accept_language_invalid(self):
        response = self.fetch('/', headers={'Accept-Language': 'fr-FR;q=-1'})
        self.assertEqual(response.headers['Content-Language'], 'en-US')