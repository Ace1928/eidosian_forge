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
class StatusReasonTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            reason = self.request.arguments.get('reason', [])
            self.set_status(int(self.get_argument('code')), reason=to_unicode(reason[0]) if reason else None)

    def get_http_client(self):
        return SimpleAsyncHTTPClient()

    def test_status(self):
        response = self.fetch('/?code=304')
        self.assertEqual(response.code, 304)
        self.assertEqual(response.reason, 'Not Modified')
        response = self.fetch('/?code=304&reason=Foo')
        self.assertEqual(response.code, 304)
        self.assertEqual(response.reason, 'Foo')
        response = self.fetch('/?code=682&reason=Bar')
        self.assertEqual(response.code, 682)
        self.assertEqual(response.reason, 'Bar')
        response = self.fetch('/?code=682')
        self.assertEqual(response.code, 682)
        self.assertEqual(response.reason, 'Unknown')