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
class DecodeArgHandler(RequestHandler):

    def decode_argument(self, value, name=None):
        if type(value) != bytes:
            raise Exception('unexpected type for value: %r' % type(value))
        if 'encoding' in self.request.arguments:
            return value.decode(to_unicode(self.request.arguments['encoding'][0]))
        else:
            return value

    def get(self, arg):

        def describe(s):
            if type(s) == bytes:
                return ['bytes', native_str(binascii.b2a_hex(s))]
            elif type(s) == unicode_type:
                return ['unicode', s]
            raise Exception('unknown type')
        self.write({'path': describe(arg), 'query': describe(self.get_argument('foo'))})