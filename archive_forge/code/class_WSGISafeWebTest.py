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
class WSGISafeWebTest(WebTestCase):
    COOKIE_SECRET = 'WebTest.COOKIE_SECRET'

    def get_app_kwargs(self):
        loader = DictLoader({'linkify.html': '{% module linkify(message) %}', 'page.html': '<html><head></head><body>\n{% for e in entries %}\n{% module Template("entry.html", entry=e) %}\n{% end %}\n</body></html>', 'entry.html': '{{ set_resources(embedded_css=".entry { margin-bottom: 1em; }",\n                 embedded_javascript="js_embed()",\n                 css_files=["/base.css", "/foo.css"],\n                 javascript_files="/common.js",\n                 html_head="<meta>",\n                 html_body=\'<script src="/analytics.js"/>\') }}\n<div class="entry">...</div>'})
        return dict(template_loader=loader, autoescape='xhtml_escape', cookie_secret=self.COOKIE_SECRET)

    def tearDown(self):
        super().tearDown()
        RequestHandler._template_loaders.clear()

    def get_handlers(self):
        urls = [url('/typecheck/(.*)', TypeCheckHandler, name='typecheck'), url('/decode_arg/(.*)', DecodeArgHandler, name='decode_arg'), url('/decode_arg_kw/(?P<arg>.*)', DecodeArgHandler), url('/linkify', LinkifyHandler), url('/uimodule_resources', UIModuleResourceHandler), url('/optional_path/(.+)?', OptionalPathHandler), url('/multi_header', MultiHeaderHandler), url('/redirect', RedirectHandler), url('/web_redirect_permanent', WebRedirectHandler, {'url': '/web_redirect_newpath'}), url('/web_redirect', WebRedirectHandler, {'url': '/web_redirect_newpath', 'permanent': False}), url('//web_redirect_double_slash', WebRedirectHandler, {'url': '/web_redirect_newpath'}), url('/header_injection', HeaderInjectionHandler), url('/get_argument', GetArgumentHandler), url('/get_arguments', GetArgumentsHandler)]
        return urls

    def fetch_json(self, *args, **kwargs):
        response = self.fetch(*args, **kwargs)
        response.rethrow()
        return json_decode(response.body)

    def test_types(self):
        cookie_value = to_unicode(create_signed_value(self.COOKIE_SECRET, 'asdf', 'qwer'))
        response = self.fetch('/typecheck/asdf?foo=bar', headers={'Cookie': 'asdf=' + cookie_value})
        data = json_decode(response.body)
        self.assertEqual(data, {})
        response = self.fetch('/typecheck/asdf?foo=bar', method='POST', headers={'Cookie': 'asdf=' + cookie_value}, body='foo=bar')

    def test_decode_argument(self):
        urls = ['/decode_arg/%C3%A9?foo=%C3%A9&encoding=utf-8', '/decode_arg/%E9?foo=%E9&encoding=latin1', '/decode_arg_kw/%E9?foo=%E9&encoding=latin1']
        for req_url in urls:
            response = self.fetch(req_url)
            response.rethrow()
            data = json_decode(response.body)
            self.assertEqual(data, {'path': ['unicode', 'é'], 'query': ['unicode', 'é']})
        response = self.fetch('/decode_arg/%C3%A9?foo=%C3%A9')
        response.rethrow()
        data = json_decode(response.body)
        self.assertEqual(data, {'path': ['bytes', 'c3a9'], 'query': ['bytes', 'c3a9']})

    def test_decode_argument_invalid_unicode(self):
        with ExpectLog(gen_log, '.*Invalid unicode.*'):
            response = self.fetch('/typecheck/invalid%FF')
            self.assertEqual(response.code, 400)
            response = self.fetch('/typecheck/invalid?foo=%FF')
            self.assertEqual(response.code, 400)

    def test_decode_argument_plus(self):
        urls = ['/decode_arg/1%20%2B%201?foo=1%20%2B%201&encoding=utf-8', '/decode_arg/1%20+%201?foo=1+%2B+1&encoding=utf-8']
        for req_url in urls:
            response = self.fetch(req_url)
            response.rethrow()
            data = json_decode(response.body)
            self.assertEqual(data, {'path': ['unicode', '1 + 1'], 'query': ['unicode', '1 + 1']})

    def test_reverse_url(self):
        self.assertEqual(self.app.reverse_url('decode_arg', 'foo'), '/decode_arg/foo')
        self.assertEqual(self.app.reverse_url('decode_arg', 42), '/decode_arg/42')
        self.assertEqual(self.app.reverse_url('decode_arg', b'\xe9'), '/decode_arg/%E9')
        self.assertEqual(self.app.reverse_url('decode_arg', 'é'), '/decode_arg/%C3%A9')
        self.assertEqual(self.app.reverse_url('decode_arg', '1 + 1'), '/decode_arg/1%20%2B%201')

    def test_uimodule_unescaped(self):
        response = self.fetch('/linkify')
        self.assertEqual(response.body, b'<a href="http://example.com">http://example.com</a>')

    def test_uimodule_resources(self):
        response = self.fetch('/uimodule_resources')
        self.assertEqual(response.body, b'<html><head><link href="/base.css" type="text/css" rel="stylesheet"/><link href="/foo.css" type="text/css" rel="stylesheet"/>\n<style type="text/css">\n.entry { margin-bottom: 1em; }\n</style>\n<meta>\n</head><body>\n\n\n<div class="entry">...</div>\n\n\n<div class="entry">...</div>\n\n<script src="/common.js" type="text/javascript"></script>\n<script type="text/javascript">\n//<![CDATA[\njs_embed()\n//]]>\n</script>\n<script src="/analytics.js"/>\n</body></html>')

    def test_optional_path(self):
        self.assertEqual(self.fetch_json('/optional_path/foo'), {'path': 'foo'})
        self.assertEqual(self.fetch_json('/optional_path/'), {'path': None})

    def test_multi_header(self):
        response = self.fetch('/multi_header')
        self.assertEqual(response.headers['x-overwrite'], '2')
        self.assertEqual(response.headers.get_list('x-multi'), ['3', '4'])

    def test_redirect(self):
        response = self.fetch('/redirect?permanent=1', follow_redirects=False)
        self.assertEqual(response.code, 301)
        response = self.fetch('/redirect?permanent=0', follow_redirects=False)
        self.assertEqual(response.code, 302)
        response = self.fetch('/redirect?status=307', follow_redirects=False)
        self.assertEqual(response.code, 307)

    def test_web_redirect(self):
        response = self.fetch('/web_redirect_permanent', follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers['Location'], '/web_redirect_newpath')
        response = self.fetch('/web_redirect', follow_redirects=False)
        self.assertEqual(response.code, 302)
        self.assertEqual(response.headers['Location'], '/web_redirect_newpath')

    def test_web_redirect_double_slash(self):
        response = self.fetch('//web_redirect_double_slash', follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers['Location'], '/web_redirect_newpath')

    def test_header_injection(self):
        response = self.fetch('/header_injection')
        self.assertEqual(response.body, b'ok')

    def test_get_argument(self):
        response = self.fetch('/get_argument?foo=bar')
        self.assertEqual(response.body, b'bar')
        response = self.fetch('/get_argument?foo=')
        self.assertEqual(response.body, b'')
        response = self.fetch('/get_argument')
        self.assertEqual(response.body, b'default')
        body = urllib.parse.urlencode(dict(foo='hello'))
        response = self.fetch('/get_argument?foo=bar', method='POST', body=body)
        self.assertEqual(response.body, b'hello')
        response = self.fetch('/get_arguments?foo=bar', method='POST', body=body)
        self.assertEqual(json_decode(response.body), dict(default=['bar', 'hello'], query=['bar'], body=['hello']))

    def test_get_query_arguments(self):
        body = urllib.parse.urlencode(dict(foo='hello'))
        response = self.fetch('/get_argument?source=query&foo=bar', method='POST', body=body)
        self.assertEqual(response.body, b'bar')
        response = self.fetch('/get_argument?source=query&foo=', method='POST', body=body)
        self.assertEqual(response.body, b'')
        response = self.fetch('/get_argument?source=query', method='POST', body=body)
        self.assertEqual(response.body, b'default')

    def test_get_body_arguments(self):
        body = urllib.parse.urlencode(dict(foo='bar'))
        response = self.fetch('/get_argument?source=body&foo=hello', method='POST', body=body)
        self.assertEqual(response.body, b'bar')
        body = urllib.parse.urlencode(dict(foo=''))
        response = self.fetch('/get_argument?source=body&foo=hello', method='POST', body=body)
        self.assertEqual(response.body, b'')
        body = urllib.parse.urlencode(dict())
        response = self.fetch('/get_argument?source=body&foo=hello', method='POST', body=body)
        self.assertEqual(response.body, b'default')

    def test_no_gzip(self):
        response = self.fetch('/get_argument')
        self.assertNotIn('Accept-Encoding', response.headers.get('Vary', ''))
        self.assertNotIn('gzip', response.headers.get('Content-Encoding', ''))