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
class XSRFTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            version = int(self.get_argument('version', '2'))
            self.settings['xsrf_cookie_version'] = version
            self.write(self.xsrf_token)

        def post(self):
            self.write('ok')

    def get_app_kwargs(self):
        return dict(xsrf_cookies=True)

    def setUp(self):
        super().setUp()
        self.xsrf_token = self.get_token()

    def get_token(self, old_token=None, version=None):
        if old_token is not None:
            headers = self.cookie_headers(old_token)
        else:
            headers = None
        response = self.fetch('/' if version is None else '/?version=%d' % version, headers=headers)
        response.rethrow()
        return native_str(response.body)

    def cookie_headers(self, token=None):
        if token is None:
            token = self.xsrf_token
        return {'Cookie': '_xsrf=' + token}

    def test_xsrf_fail_no_token(self):
        with ExpectLog(gen_log, ".*'_xsrf' argument missing"):
            response = self.fetch('/', method='POST', body=b'')
        self.assertEqual(response.code, 403)

    def test_xsrf_fail_body_no_cookie(self):
        with ExpectLog(gen_log, '.*XSRF cookie does not match POST'):
            response = self.fetch('/', method='POST', body=urllib.parse.urlencode(dict(_xsrf=self.xsrf_token)))
        self.assertEqual(response.code, 403)

    def test_xsrf_fail_argument_invalid_format(self):
        with ExpectLog(gen_log, ".*'_xsrf' argument has invalid format"):
            response = self.fetch('/', method='POST', headers=self.cookie_headers(), body=urllib.parse.urlencode(dict(_xsrf='3|')))
        self.assertEqual(response.code, 403)

    def test_xsrf_fail_cookie_invalid_format(self):
        with ExpectLog(gen_log, '.*XSRF cookie does not match POST'):
            response = self.fetch('/', method='POST', headers=self.cookie_headers(token='3|'), body=urllib.parse.urlencode(dict(_xsrf=self.xsrf_token)))
        self.assertEqual(response.code, 403)

    def test_xsrf_fail_cookie_no_body(self):
        with ExpectLog(gen_log, ".*'_xsrf' argument missing"):
            response = self.fetch('/', method='POST', body=b'', headers=self.cookie_headers())
        self.assertEqual(response.code, 403)

    def test_xsrf_success_short_token(self):
        response = self.fetch('/', method='POST', body=urllib.parse.urlencode(dict(_xsrf='deadbeef')), headers=self.cookie_headers(token='deadbeef'))
        self.assertEqual(response.code, 200)

    def test_xsrf_success_non_hex_token(self):
        response = self.fetch('/', method='POST', body=urllib.parse.urlencode(dict(_xsrf='xoxo')), headers=self.cookie_headers(token='xoxo'))
        self.assertEqual(response.code, 200)

    def test_xsrf_success_post_body(self):
        response = self.fetch('/', method='POST', body=urllib.parse.urlencode(dict(_xsrf=self.xsrf_token)), headers=self.cookie_headers())
        self.assertEqual(response.code, 200)

    def test_xsrf_success_query_string(self):
        response = self.fetch('/?' + urllib.parse.urlencode(dict(_xsrf=self.xsrf_token)), method='POST', body=b'', headers=self.cookie_headers())
        self.assertEqual(response.code, 200)

    def test_xsrf_success_header(self):
        response = self.fetch('/', method='POST', body=b'', headers=dict({'X-Xsrftoken': self.xsrf_token}, **self.cookie_headers()))
        self.assertEqual(response.code, 200)

    def test_distinct_tokens(self):
        NUM_TOKENS = 10
        tokens = set()
        for i in range(NUM_TOKENS):
            tokens.add(self.get_token())
        self.assertEqual(len(tokens), NUM_TOKENS)

    def test_cross_user(self):
        token2 = self.get_token()
        for token in (self.xsrf_token, token2):
            response = self.fetch('/', method='POST', body=urllib.parse.urlencode(dict(_xsrf=token)), headers=self.cookie_headers(token))
            self.assertEqual(response.code, 200)
        for cookie_token, body_token in ((self.xsrf_token, token2), (token2, self.xsrf_token)):
            with ExpectLog(gen_log, '.*XSRF cookie does not match POST'):
                response = self.fetch('/', method='POST', body=urllib.parse.urlencode(dict(_xsrf=body_token)), headers=self.cookie_headers(cookie_token))
            self.assertEqual(response.code, 403)

    def test_refresh_token(self):
        token = self.xsrf_token
        tokens_seen = set([token])
        for i in range(5):
            token = self.get_token(token)
            tokens_seen.add(token)
            response = self.fetch('/', method='POST', body=urllib.parse.urlencode(dict(_xsrf=self.xsrf_token)), headers=self.cookie_headers(token))
            self.assertEqual(response.code, 200)
        self.assertEqual(len(tokens_seen), 6)

    def test_versioning(self):
        self.assertNotEqual(self.get_token(version=1), self.get_token(version=1))
        v1_token = self.get_token(version=1)
        for i in range(5):
            self.assertEqual(self.get_token(v1_token, version=1), v1_token)
        v2_token = self.get_token(v1_token)
        self.assertNotEqual(v1_token, v2_token)
        self.assertNotEqual(v2_token, self.get_token(v1_token))
        for cookie_token, body_token in ((v1_token, v2_token), (v2_token, v1_token)):
            response = self.fetch('/', method='POST', body=urllib.parse.urlencode(dict(_xsrf=body_token)), headers=self.cookie_headers(cookie_token))
            self.assertEqual(response.code, 200)