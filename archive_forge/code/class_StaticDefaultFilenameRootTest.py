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
class StaticDefaultFilenameRootTest(WebTestCase):

    def get_app_kwargs(self):
        return dict(static_path=os.path.abspath(relpath('static')), static_handler_args=dict(default_filename='index.html'), static_url_prefix='/')

    def get_handlers(self):
        return []

    def get_http_client(self):
        return SimpleAsyncHTTPClient()

    def test_no_open_redirect(self):
        test_dir = os.path.dirname(__file__)
        drive, tail = os.path.splitdrive(test_dir)
        if os.name == 'posix':
            self.assertEqual(tail, test_dir)
        else:
            test_dir = tail
        with ExpectLog(gen_log, '.*cannot redirect path with two initial slashes'):
            response = self.fetch(f'//evil.com/../{test_dir}/static/dir', follow_redirects=False)
        self.assertEqual(response.code, 403)