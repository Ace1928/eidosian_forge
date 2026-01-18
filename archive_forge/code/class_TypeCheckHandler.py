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
class TypeCheckHandler(RequestHandler):

    def prepare(self):
        self.errors = {}
        self.check_type('status', self.get_status(), int)
        self.check_type('argument', self.get_argument('foo'), unicode_type)
        self.check_type('cookie_key', list(self.cookies.keys())[0], str)
        self.check_type('cookie_value', list(self.cookies.values())[0].value, str)
        if list(self.cookies.keys()) != ['asdf']:
            raise Exception('unexpected values for cookie keys: %r' % self.cookies.keys())
        self.check_type('get_signed_cookie', self.get_signed_cookie('asdf'), bytes)
        self.check_type('get_cookie', self.get_cookie('asdf'), str)
        self.check_type('xsrf_token', self.xsrf_token, bytes)
        self.check_type('xsrf_form_html', self.xsrf_form_html(), str)
        self.check_type('reverse_url', self.reverse_url('typecheck', 'foo'), str)
        self.check_type('request_summary', self._request_summary(), str)

    def get(self, path_component):
        self.check_type('path_component', path_component, unicode_type)
        self.write(self.errors)

    def post(self, path_component):
        self.check_type('path_component', path_component, unicode_type)
        self.write(self.errors)

    def check_type(self, name, obj, expected_type):
        actual_type = type(obj)
        if expected_type != actual_type:
            self.errors[name] = 'expected %s, got %s' % (expected_type, actual_type)