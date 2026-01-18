from tornado import gen, netutil
from tornado.escape import (
from tornado.http1connection import HTTP1Connection
from tornado.httpclient import HTTPError
from tornado.httpserver import HTTPServer
from tornado.httputil import (
from tornado.iostream import IOStream
from tornado.locks import Event
from tornado.log import gen_log, app_log
from tornado.netutil import ssl_options_to_context
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.testing import (
from tornado.test.util import skipOnTravis
from tornado.web import Application, RequestHandler, stream_request_body
from contextlib import closing
import datetime
import gzip
import logging
import os
import shutil
import socket
import ssl
import sys
import tempfile
import textwrap
import unittest
import urllib.parse
from io import BytesIO
import typing
class XHeaderTest(HandlerBaseTestCase):

    class Handler(RequestHandler):

        def get(self):
            self.set_header('request-version', self.request.version)
            self.write(dict(remote_ip=self.request.remote_ip, remote_protocol=self.request.protocol))

    def get_httpserver_options(self):
        return dict(xheaders=True, trusted_downstream=['5.5.5.5'])

    def test_ip_headers(self):
        self.assertEqual(self.fetch_json('/')['remote_ip'], '127.0.0.1')
        valid_ipv4 = {'X-Real-IP': '4.4.4.4'}
        self.assertEqual(self.fetch_json('/', headers=valid_ipv4)['remote_ip'], '4.4.4.4')
        valid_ipv4_list = {'X-Forwarded-For': '127.0.0.1, 4.4.4.4'}
        self.assertEqual(self.fetch_json('/', headers=valid_ipv4_list)['remote_ip'], '4.4.4.4')
        valid_ipv6 = {'X-Real-IP': '2620:0:1cfe:face:b00c::3'}
        self.assertEqual(self.fetch_json('/', headers=valid_ipv6)['remote_ip'], '2620:0:1cfe:face:b00c::3')
        valid_ipv6_list = {'X-Forwarded-For': '::1, 2620:0:1cfe:face:b00c::3'}
        self.assertEqual(self.fetch_json('/', headers=valid_ipv6_list)['remote_ip'], '2620:0:1cfe:face:b00c::3')
        invalid_chars = {'X-Real-IP': '4.4.4.4<script>'}
        self.assertEqual(self.fetch_json('/', headers=invalid_chars)['remote_ip'], '127.0.0.1')
        invalid_chars_list = {'X-Forwarded-For': '4.4.4.4, 5.5.5.5<script>'}
        self.assertEqual(self.fetch_json('/', headers=invalid_chars_list)['remote_ip'], '127.0.0.1')
        invalid_host = {'X-Real-IP': 'www.google.com'}
        self.assertEqual(self.fetch_json('/', headers=invalid_host)['remote_ip'], '127.0.0.1')

    def test_trusted_downstream(self):
        valid_ipv4_list = {'X-Forwarded-For': '127.0.0.1, 4.4.4.4, 5.5.5.5'}
        resp = self.fetch('/', headers=valid_ipv4_list)
        if resp.headers['request-version'].startswith('HTTP/2'):
            self.skipTest('requires HTTP/1.x')
        result = json_decode(resp.body)
        self.assertEqual(result['remote_ip'], '4.4.4.4')

    def test_scheme_headers(self):
        self.assertEqual(self.fetch_json('/')['remote_protocol'], 'http')
        https_scheme = {'X-Scheme': 'https'}
        self.assertEqual(self.fetch_json('/', headers=https_scheme)['remote_protocol'], 'https')
        https_forwarded = {'X-Forwarded-Proto': 'https'}
        self.assertEqual(self.fetch_json('/', headers=https_forwarded)['remote_protocol'], 'https')
        https_multi_forwarded = {'X-Forwarded-Proto': 'https , http'}
        self.assertEqual(self.fetch_json('/', headers=https_multi_forwarded)['remote_protocol'], 'http')
        http_multi_forwarded = {'X-Forwarded-Proto': 'http,https'}
        self.assertEqual(self.fetch_json('/', headers=http_multi_forwarded)['remote_protocol'], 'https')
        bad_forwarded = {'X-Forwarded-Proto': 'unknown'}
        self.assertEqual(self.fetch_json('/', headers=bad_forwarded)['remote_protocol'], 'http')