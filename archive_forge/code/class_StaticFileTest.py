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
class StaticFileTest(WebTestCase):
    robots_txt_hash = b'63a36e950e134b5217e33c763e88840c10a07d80e6057d92b9ac97508de7fb1fa6f0e9b7531e169657165ea764e8963399cb6d921ffe6078425aaafe54c04563'
    static_dir = os.path.join(os.path.dirname(__file__), 'static')

    def get_handlers(self):

        class StaticUrlHandler(RequestHandler):

            def get(self, path):
                with_v = int(self.get_argument('include_version', '1'))
                self.write(self.static_url(path, include_version=with_v))

        class AbsoluteStaticUrlHandler(StaticUrlHandler):
            include_host = True

        class OverrideStaticUrlHandler(RequestHandler):

            def get(self, path):
                do_include = bool(self.get_argument('include_host'))
                self.include_host = not do_include
                regular_url = self.static_url(path)
                override_url = self.static_url(path, include_host=do_include)
                if override_url == regular_url:
                    return self.write(str(False))
                protocol = self.request.protocol + '://'
                protocol_length = len(protocol)
                check_regular = regular_url.find(protocol, 0, protocol_length)
                check_override = override_url.find(protocol, 0, protocol_length)
                if do_include:
                    result = check_override == 0 and check_regular == -1
                else:
                    result = check_override == -1 and check_regular == 0
                self.write(str(result))
        return [('/static_url/(.*)', StaticUrlHandler), ('/abs_static_url/(.*)', AbsoluteStaticUrlHandler), ('/override_static_url/(.*)', OverrideStaticUrlHandler), ('/root_static/(.*)', StaticFileHandler, dict(path='/'))]

    def get_app_kwargs(self):
        return dict(static_path=relpath('static'))

    def test_static_files(self):
        response = self.fetch('/robots.txt')
        self.assertTrue(b'Disallow: /' in response.body)
        response = self.fetch('/static/robots.txt')
        self.assertTrue(b'Disallow: /' in response.body)
        self.assertEqual(response.headers.get('Content-Type'), 'text/plain')

    def test_static_files_cacheable(self):
        response = self.fetch('/robots.txt?v=12345')
        self.assertTrue(b'Disallow: /' in response.body)
        self.assertIn('Cache-Control', response.headers)
        self.assertIn('Expires', response.headers)

    def test_static_compressed_files(self):
        response = self.fetch('/static/sample.xml.gz')
        self.assertEqual(response.headers.get('Content-Type'), 'application/gzip')
        response = self.fetch('/static/sample.xml.bz2')
        self.assertEqual(response.headers.get('Content-Type'), 'application/octet-stream')
        response = self.fetch('/static/sample.xml')
        self.assertTrue(response.headers.get('Content-Type') in set(('text/xml', 'application/xml')))

    def test_static_url(self):
        response = self.fetch('/static_url/robots.txt')
        self.assertEqual(response.body, b'/static/robots.txt?v=' + self.robots_txt_hash)

    def test_absolute_static_url(self):
        response = self.fetch('/abs_static_url/robots.txt')
        self.assertEqual(response.body, utf8(self.get_url('/')) + b'static/robots.txt?v=' + self.robots_txt_hash)

    def test_relative_version_exclusion(self):
        response = self.fetch('/static_url/robots.txt?include_version=0')
        self.assertEqual(response.body, b'/static/robots.txt')

    def test_absolute_version_exclusion(self):
        response = self.fetch('/abs_static_url/robots.txt?include_version=0')
        self.assertEqual(response.body, utf8(self.get_url('/') + 'static/robots.txt'))

    def test_include_host_override(self):
        self._trigger_include_host_check(False)
        self._trigger_include_host_check(True)

    def _trigger_include_host_check(self, include_host):
        path = '/override_static_url/robots.txt?include_host=%s'
        response = self.fetch(path % int(include_host))
        self.assertEqual(response.body, utf8(str(True)))

    def get_and_head(self, *args, **kwargs):
        """Performs a GET and HEAD request and returns the GET response.

        Fails if any ``Content-*`` headers returned by the two requests
        differ.
        """
        head_response = self.fetch(*args, method='HEAD', **kwargs)
        get_response = self.fetch(*args, method='GET', **kwargs)
        content_headers = set()
        for h in itertools.chain(head_response.headers, get_response.headers):
            if h.startswith('Content-'):
                content_headers.add(h)
        for h in content_headers:
            self.assertEqual(head_response.headers.get(h), get_response.headers.get(h), '%s differs between GET (%s) and HEAD (%s)' % (h, head_response.headers.get(h), get_response.headers.get(h)))
        return get_response

    def test_static_304_if_modified_since(self):
        response1 = self.get_and_head('/static/robots.txt')
        response2 = self.get_and_head('/static/robots.txt', headers={'If-Modified-Since': response1.headers['Last-Modified']})
        self.assertEqual(response2.code, 304)
        self.assertTrue('Content-Length' not in response2.headers)

    def test_static_304_if_none_match(self):
        response1 = self.get_and_head('/static/robots.txt')
        response2 = self.get_and_head('/static/robots.txt', headers={'If-None-Match': response1.headers['Etag']})
        self.assertEqual(response2.code, 304)

    def test_static_304_etag_modified_bug(self):
        response1 = self.get_and_head('/static/robots.txt')
        response2 = self.get_and_head('/static/robots.txt', headers={'If-None-Match': '"MISMATCH"', 'If-Modified-Since': response1.headers['Last-Modified']})
        self.assertEqual(response2.code, 200)

    def test_static_if_modified_since_pre_epoch(self):
        response = self.get_and_head('/static/robots.txt', headers={'If-Modified-Since': 'Fri, 01 Jan 1960 00:00:00 GMT'})
        self.assertEqual(response.code, 200)

    def test_static_if_modified_since_time_zone(self):
        stat = os.stat(relpath('static/robots.txt'))
        response = self.get_and_head('/static/robots.txt', headers={'If-Modified-Since': format_timestamp(stat.st_mtime - 1)})
        self.assertEqual(response.code, 200)
        response = self.get_and_head('/static/robots.txt', headers={'If-Modified-Since': format_timestamp(stat.st_mtime + 1)})
        self.assertEqual(response.code, 304)

    def test_static_etag(self):
        response = self.get_and_head('/static/robots.txt')
        self.assertEqual(utf8(response.headers.get('Etag')), b'"' + self.robots_txt_hash + b'"')

    def test_static_with_range(self):
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=0-9'})
        self.assertEqual(response.code, 206)
        self.assertEqual(response.body, b'User-agent')
        self.assertEqual(utf8(response.headers.get('Etag')), b'"' + self.robots_txt_hash + b'"')
        self.assertEqual(response.headers.get('Content-Length'), '10')
        self.assertEqual(response.headers.get('Content-Range'), 'bytes 0-9/26')

    def test_static_with_range_full_file(self):
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=0-'})
        self.assertEqual(response.code, 200)
        robots_file_path = os.path.join(self.static_dir, 'robots.txt')
        with open(robots_file_path, encoding='utf-8') as f:
            self.assertEqual(response.body, utf8(f.read()))
        self.assertEqual(response.headers.get('Content-Length'), '26')
        self.assertEqual(response.headers.get('Content-Range'), None)

    def test_static_with_range_full_past_end(self):
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=0-10000000'})
        self.assertEqual(response.code, 200)
        robots_file_path = os.path.join(self.static_dir, 'robots.txt')
        with open(robots_file_path, encoding='utf-8') as f:
            self.assertEqual(response.body, utf8(f.read()))
        self.assertEqual(response.headers.get('Content-Length'), '26')
        self.assertEqual(response.headers.get('Content-Range'), None)

    def test_static_with_range_partial_past_end(self):
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=1-10000000'})
        self.assertEqual(response.code, 206)
        robots_file_path = os.path.join(self.static_dir, 'robots.txt')
        with open(robots_file_path, encoding='utf-8') as f:
            self.assertEqual(response.body, utf8(f.read()[1:]))
        self.assertEqual(response.headers.get('Content-Length'), '25')
        self.assertEqual(response.headers.get('Content-Range'), 'bytes 1-25/26')

    def test_static_with_range_end_edge(self):
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=22-'})
        self.assertEqual(response.body, b': /\n')
        self.assertEqual(response.headers.get('Content-Length'), '4')
        self.assertEqual(response.headers.get('Content-Range'), 'bytes 22-25/26')

    def test_static_with_range_neg_end(self):
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=-4'})
        self.assertEqual(response.body, b': /\n')
        self.assertEqual(response.headers.get('Content-Length'), '4')
        self.assertEqual(response.headers.get('Content-Range'), 'bytes 22-25/26')

    def test_static_with_range_neg_past_start(self):
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=-1000000'})
        self.assertEqual(response.code, 200)
        robots_file_path = os.path.join(self.static_dir, 'robots.txt')
        with open(robots_file_path, encoding='utf-8') as f:
            self.assertEqual(response.body, utf8(f.read()))
        self.assertEqual(response.headers.get('Content-Length'), '26')
        self.assertEqual(response.headers.get('Content-Range'), None)

    def test_static_invalid_range(self):
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'asdf'})
        self.assertEqual(response.code, 200)

    def test_static_unsatisfiable_range_zero_suffix(self):
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=-0'})
        self.assertEqual(response.headers.get('Content-Range'), 'bytes */26')
        self.assertEqual(response.code, 416)

    def test_static_unsatisfiable_range_invalid_start(self):
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=26'})
        self.assertEqual(response.code, 416)
        self.assertEqual(response.headers.get('Content-Range'), 'bytes */26')

    def test_static_unsatisfiable_range_end_less_than_start(self):
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=10-3'})
        self.assertEqual(response.code, 416)
        self.assertEqual(response.headers.get('Content-Range'), 'bytes */26')

    def test_static_head(self):
        response = self.fetch('/static/robots.txt', method='HEAD')
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body, b'')
        self.assertEqual(response.headers['Content-Length'], '26')
        self.assertEqual(utf8(response.headers['Etag']), b'"' + self.robots_txt_hash + b'"')

    def test_static_head_range(self):
        response = self.fetch('/static/robots.txt', method='HEAD', headers={'Range': 'bytes=1-4'})
        self.assertEqual(response.code, 206)
        self.assertEqual(response.body, b'')
        self.assertEqual(response.headers['Content-Length'], '4')
        self.assertEqual(utf8(response.headers['Etag']), b'"' + self.robots_txt_hash + b'"')

    def test_static_range_if_none_match(self):
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=1-4', 'If-None-Match': b'"' + self.robots_txt_hash + b'"'})
        self.assertEqual(response.code, 304)
        self.assertEqual(response.body, b'')
        self.assertTrue('Content-Length' not in response.headers)
        self.assertEqual(utf8(response.headers['Etag']), b'"' + self.robots_txt_hash + b'"')

    def test_static_404(self):
        response = self.get_and_head('/static/blarg')
        self.assertEqual(response.code, 404)

    def test_path_traversal_protection(self):
        self.http_client.close()
        self.http_client = SimpleAsyncHTTPClient()
        with ExpectLog(gen_log, '.*not in root static directory'):
            response = self.get_and_head('/static/../static_foo.txt')
        self.assertEqual(response.code, 403)

    @unittest.skipIf(os.name != 'posix', 'non-posix OS')
    def test_root_static_path(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/robots.txt')
        response = self.get_and_head('/root_static' + urllib.parse.quote(path))
        self.assertEqual(response.code, 200)