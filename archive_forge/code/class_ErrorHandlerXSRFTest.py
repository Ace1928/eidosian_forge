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
class ErrorHandlerXSRFTest(WebTestCase):

    def get_handlers(self):
        return [('/error', ErrorHandler, dict(status_code=417))]

    def get_app_kwargs(self):
        return dict(xsrf_cookies=True)

    def test_error_xsrf(self):
        response = self.fetch('/error', method='POST', body='')
        self.assertEqual(response.code, 417)

    def test_404_xsrf(self):
        response = self.fetch('/404', method='POST', body='')
        self.assertEqual(response.code, 404)