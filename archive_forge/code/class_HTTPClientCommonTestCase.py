import base64
import binascii
from contextlib import closing
import copy
import gzip
import threading
import datetime
from io import BytesIO
import subprocess
import sys
import time
import typing  # noqa: F401
import unicodedata
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado import gen
from tornado.httpclient import (
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.iostream import IOStream
from tornado.log import gen_log, app_log
from tornado import netutil
from tornado.testing import AsyncHTTPTestCase, bind_unused_port, gen_test, ExpectLog
from tornado.test.util import skipOnTravis, ignore_deprecation
from tornado.web import Application, RequestHandler, url
from tornado.httputil import format_timestamp, HTTPHeaders
class HTTPClientCommonTestCase(AsyncHTTPTestCase):

    def get_app(self):
        return Application([url('/hello', HelloWorldHandler), url('/post', PostHandler), url('/put', PutHandler), url('/redirect', RedirectHandler), url('/redirect_without_location', RedirectWithoutLocationHandler), url('/chunk', ChunkHandler), url('/auth', AuthHandler), url('/countdown/([0-9]+)', CountdownHandler, name='countdown'), url('/echopost', EchoPostHandler), url('/user_agent', UserAgentHandler), url('/304_with_content_length', ContentLength304Handler), url('/all_methods', AllMethodsHandler), url('/patch', PatchHandler), url('/set_header', SetHeaderHandler), url('/invalid_gzip', InvalidGzipHandler), url('/header-encoding', HeaderEncodingHandler)], gzip=True)

    def test_patch_receives_payload(self):
        body = b'some patch data'
        response = self.fetch('/patch', method='PATCH', body=body)
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body, body)

    @skipOnTravis
    def test_hello_world(self):
        response = self.fetch('/hello')
        self.assertEqual(response.code, 200)
        self.assertEqual(response.headers['Content-Type'], 'text/plain')
        self.assertEqual(response.body, b'Hello world!')
        assert response.request_time is not None
        self.assertEqual(int(response.request_time), 0)
        response = self.fetch('/hello?name=Ben')
        self.assertEqual(response.body, b'Hello Ben!')

    def test_streaming_callback(self):
        chunks = []
        response = self.fetch('/hello', streaming_callback=chunks.append)
        self.assertEqual(chunks, [b'Hello world!'])
        self.assertFalse(response.body)

    def test_post(self):
        response = self.fetch('/post', method='POST', body='arg1=foo&arg2=bar')
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body, b'Post arg1: foo, arg2: bar')

    def test_chunked(self):
        response = self.fetch('/chunk')
        self.assertEqual(response.body, b'asdfqwer')
        chunks = []
        response = self.fetch('/chunk', streaming_callback=chunks.append)
        self.assertEqual(chunks, [b'asdf', b'qwer'])
        self.assertFalse(response.body)

    def test_chunked_close(self):
        sock, port = bind_unused_port()
        with closing(sock):

            @gen.coroutine
            def accept_callback(conn, address):
                stream = IOStream(conn)
                request_data = (yield stream.read_until(b'\r\n\r\n'))
                if b'HTTP/1.' not in request_data:
                    self.skipTest('requires HTTP/1.x')
                yield stream.write(b'HTTP/1.1 200 OK\nTransfer-Encoding: chunked\n\n1\n1\n1\n2\n0\n\n'.replace(b'\n', b'\r\n'))
                stream.close()
            netutil.add_accept_handler(sock, accept_callback)
            resp = self.fetch('http://127.0.0.1:%d/' % port)
            resp.rethrow()
            self.assertEqual(resp.body, b'12')
            self.io_loop.remove_handler(sock.fileno())

    def test_basic_auth(self):
        self.assertEqual(self.fetch('/auth', auth_username='Aladdin', auth_password='open sesame').body, b'Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==')

    def test_basic_auth_explicit_mode(self):
        self.assertEqual(self.fetch('/auth', auth_username='Aladdin', auth_password='open sesame', auth_mode='basic').body, b'Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==')

    def test_basic_auth_unicode(self):
        self.assertEqual(self.fetch('/auth', auth_username='test', auth_password='123£').body, b'Basic dGVzdDoxMjPCow==')
        username = unicodedata.normalize('NFD', 'josé')
        self.assertEqual(self.fetch('/auth', auth_username=username, auth_password='səcrət').body, b'Basic am9zw6k6c8mZY3LJmXQ=')

    def test_unsupported_auth_mode(self):
        with ExpectLog(gen_log, 'uncaught exception', required=False):
            with self.assertRaises((ValueError, HTTPError)):
                self.fetch('/auth', auth_username='Aladdin', auth_password='open sesame', auth_mode='asdf', raise_error=True)

    def test_follow_redirect(self):
        response = self.fetch('/countdown/2', follow_redirects=False)
        self.assertEqual(302, response.code)
        self.assertTrue(response.headers['Location'].endswith('/countdown/1'))
        response = self.fetch('/countdown/2')
        self.assertEqual(200, response.code)
        self.assertTrue(response.effective_url.endswith('/countdown/0'))
        self.assertEqual(b'Zero', response.body)

    def test_redirect_without_location(self):
        response = self.fetch('/redirect_without_location', follow_redirects=True)
        self.assertEqual(301, response.code)

    def test_redirect_put_with_body(self):
        response = self.fetch('/redirect?url=/put&status=307', method='PUT', body='hello')
        self.assertEqual(response.body, b'Put body: hello')

    def test_redirect_put_without_body(self):
        response = self.fetch('/redirect?url=/put&status=307', method='PUT', allow_nonstandard_methods=True)
        self.assertEqual(response.body, b'Put body: ')

    def test_method_after_redirect(self):
        for status in [301, 302, 303]:
            url = '/redirect?url=/all_methods&status=%d' % status
            resp = self.fetch(url, method='POST', body=b'')
            self.assertEqual(b'GET', resp.body)
            for method in ['GET', 'OPTIONS', 'PUT', 'DELETE']:
                resp = self.fetch(url, method=method, allow_nonstandard_methods=True)
                if status in [301, 302]:
                    self.assertEqual(utf8(method), resp.body)
                else:
                    self.assertIn(resp.body, [utf8(method), b'GET'])
            resp = self.fetch(url, method='HEAD')
            self.assertEqual(200, resp.code)
            self.assertEqual(b'', resp.body)
        for status in [307, 308]:
            url = '/redirect?url=/all_methods&status=307'
            for method in ['GET', 'OPTIONS', 'POST', 'PUT', 'DELETE']:
                resp = self.fetch(url, method=method, allow_nonstandard_methods=True)
                self.assertEqual(method, to_unicode(resp.body))
            resp = self.fetch(url, method='HEAD')
            self.assertEqual(200, resp.code)
            self.assertEqual(b'', resp.body)

    def test_credentials_in_url(self):
        url = self.get_url('/auth').replace('http://', 'http://me:secret@')
        response = self.fetch(url)
        self.assertEqual(b'Basic ' + base64.b64encode(b'me:secret'), response.body)

    def test_body_encoding(self):
        unicode_body = 'é'
        byte_body = binascii.a2b_hex(b'e9')
        response = self.fetch('/echopost', method='POST', body=unicode_body, headers={'Content-Type': 'application/blah'})
        self.assertEqual(response.headers['Content-Length'], '2')
        self.assertEqual(response.body, utf8(unicode_body))
        response = self.fetch('/echopost', method='POST', body=byte_body, headers={'Content-Type': 'application/blah'})
        self.assertEqual(response.headers['Content-Length'], '1')
        self.assertEqual(response.body, byte_body)
        response = self.fetch('/echopost', method='POST', body=byte_body, headers={'Content-Type': 'application/blah'}, user_agent='foo')
        self.assertEqual(response.headers['Content-Length'], '1')
        self.assertEqual(response.body, byte_body)

    def test_types(self):
        response = self.fetch('/hello')
        self.assertEqual(type(response.body), bytes)
        self.assertEqual(type(response.headers['Content-Type']), str)
        self.assertEqual(type(response.code), int)
        self.assertEqual(type(response.effective_url), str)

    def test_gzip(self):
        response = self.fetch('/chunk', decompress_response=False, headers={'Accept-Encoding': 'gzip'})
        self.assertEqual(response.headers['Content-Encoding'], 'gzip')
        self.assertNotEqual(response.body, b'asdfqwer')
        self.assertEqual(len(response.body), 34)
        f = gzip.GzipFile(mode='r', fileobj=response.buffer)
        self.assertEqual(f.read(), b'asdfqwer')

    def test_invalid_gzip(self):
        with ExpectLog(app_log, '(Uncaught exception|Exception in callback)', required=False):
            try:
                response = self.fetch('/invalid_gzip')
                self.assertEqual(response.code, 200)
                self.assertEqual(response.body[:14], b'Hello World 0\n')
            except HTTPError:
                pass

    def test_header_callback(self):
        first_line = []
        headers = {}
        chunks = []

        def header_callback(header_line):
            if header_line.startswith('HTTP/1.1 101'):
                pass
            elif header_line.startswith('HTTP/'):
                first_line.append(header_line)
            elif header_line != '\r\n':
                k, v = header_line.split(':', 1)
                headers[k.lower()] = v.strip()

        def streaming_callback(chunk):
            self.assertEqual(headers['content-type'], 'text/html; charset=UTF-8')
            chunks.append(chunk)
        self.fetch('/chunk', header_callback=header_callback, streaming_callback=streaming_callback)
        self.assertEqual(len(first_line), 1, first_line)
        self.assertRegex(first_line[0], 'HTTP/[0-9]\\.[0-9] 200.*\r\n')
        self.assertEqual(chunks, [b'asdf', b'qwer'])

    @gen_test
    def test_configure_defaults(self):
        defaults = dict(user_agent='TestDefaultUserAgent', allow_ipv6=False)
        client = self.http_client.__class__(force_instance=True, defaults=defaults)
        try:
            response = (yield client.fetch(self.get_url('/user_agent')))
            self.assertEqual(response.body, b'TestDefaultUserAgent')
        finally:
            client.close()

    def test_header_types(self):
        for value in ['MyUserAgent', b'MyUserAgent']:
            for container in [dict, HTTPHeaders]:
                headers = container()
                headers['User-Agent'] = value
                resp = self.fetch('/user_agent', headers=headers)
                self.assertEqual(resp.body, b'MyUserAgent', 'response=%r, value=%r, container=%r' % (resp.body, value, container))

    def test_multi_line_headers(self):
        sock, port = bind_unused_port()
        with closing(sock):

            @gen.coroutine
            def accept_callback(conn, address):
                stream = IOStream(conn)
                request_data = (yield stream.read_until(b'\r\n\r\n'))
                if b'HTTP/1.' not in request_data:
                    self.skipTest('requires HTTP/1.x')
                yield stream.write(b'HTTP/1.1 200 OK\nX-XSS-Protection: 1;\n\tmode=block\n\n'.replace(b'\n', b'\r\n'))
                stream.close()
            netutil.add_accept_handler(sock, accept_callback)
            try:
                resp = self.fetch('http://127.0.0.1:%d/' % port)
                resp.rethrow()
                self.assertEqual(resp.headers['X-XSS-Protection'], '1; mode=block')
            finally:
                self.io_loop.remove_handler(sock.fileno())

    @gen_test
    def test_header_encoding(self):
        response = (yield self.http_client.fetch(self.get_url('/header-encoding'), headers={'Foo': 'bär'}))
        self.assertEqual(response.body, 'bär'.encode('ISO8859-1'))

    def test_304_with_content_length(self):
        response = self.fetch('/304_with_content_length')
        self.assertEqual(response.code, 304)
        self.assertEqual(response.headers['Content-Length'], '42')

    @gen_test
    def test_future_interface(self):
        response = (yield self.http_client.fetch(self.get_url('/hello')))
        self.assertEqual(response.body, b'Hello world!')

    @gen_test
    def test_future_http_error(self):
        with self.assertRaises(HTTPError) as context:
            yield self.http_client.fetch(self.get_url('/notfound'))
        assert context.exception is not None
        assert context.exception.response is not None
        self.assertEqual(context.exception.code, 404)
        self.assertEqual(context.exception.response.code, 404)

    @gen_test
    def test_future_http_error_no_raise(self):
        response = (yield self.http_client.fetch(self.get_url('/notfound'), raise_error=False))
        self.assertEqual(response.code, 404)

    @gen_test
    def test_reuse_request_from_response(self):
        url = self.get_url('/hello')
        response = (yield self.http_client.fetch(url))
        self.assertEqual(response.request.url, url)
        self.assertTrue(isinstance(response.request, HTTPRequest))
        response2 = (yield self.http_client.fetch(response.request))
        self.assertEqual(response2.body, b'Hello world!')

    @gen_test
    def test_bind_source_ip(self):
        url = self.get_url('/hello')
        request = HTTPRequest(url, network_interface='127.0.0.1')
        response = (yield self.http_client.fetch(request))
        self.assertEqual(response.code, 200)
        with self.assertRaises((ValueError, HTTPError)) as context:
            request = HTTPRequest(url, network_interface='not-interface-or-ip')
            yield self.http_client.fetch(request)
        self.assertIn('not-interface-or-ip', str(context.exception))

    def test_all_methods(self):
        for method in ['GET', 'DELETE', 'OPTIONS']:
            response = self.fetch('/all_methods', method=method)
            self.assertEqual(response.body, utf8(method))
        for method in ['POST', 'PUT', 'PATCH']:
            response = self.fetch('/all_methods', method=method, body=b'')
            self.assertEqual(response.body, utf8(method))
        response = self.fetch('/all_methods', method='HEAD')
        self.assertEqual(response.body, b'')
        response = self.fetch('/all_methods', method='OTHER', allow_nonstandard_methods=True)
        self.assertEqual(response.body, b'OTHER')

    def test_body_sanity_checks(self):
        for method in ('POST', 'PUT', 'PATCH'):
            with self.assertRaises(ValueError) as context:
                self.fetch('/all_methods', method=method, raise_error=True)
            self.assertIn('must not be None', str(context.exception))
            resp = self.fetch('/all_methods', method=method, allow_nonstandard_methods=True)
            self.assertEqual(resp.code, 200)
        for method in ('GET', 'DELETE', 'OPTIONS'):
            with self.assertRaises(ValueError) as context:
                self.fetch('/all_methods', method=method, body=b'asdf', raise_error=True)
            self.assertIn('must be None', str(context.exception))
            if method != 'GET':
                self.fetch('/all_methods', method=method, body=b'asdf', allow_nonstandard_methods=True, raise_error=True)
                self.assertEqual(resp.code, 200)

    def test_put_307(self):
        response = self.fetch('/redirect?status=307&url=/put', method='PUT', body=b'hello')
        response.rethrow()
        self.assertEqual(response.body, b'Put body: hello')

    def test_non_ascii_header(self):
        response = self.fetch('/set_header?k=foo&v=%E9')
        response.rethrow()
        self.assertEqual(response.headers['Foo'], native_str('é'))

    def test_response_times(self):
        start_time = time.time()
        response = self.fetch('/hello')
        response.rethrow()
        assert response.request_time is not None
        self.assertGreaterEqual(response.request_time, 0)
        self.assertLess(response.request_time, 1.0)
        assert response.start_time is not None
        self.assertLess(abs(response.start_time - start_time), 1.0)
        for k, v in response.time_info.items():
            self.assertTrue(0 <= v < 1.0, 'time_info[%s] out of bounds: %s' % (k, v))

    def test_zero_timeout(self):
        response = self.fetch('/hello', connect_timeout=0)
        self.assertEqual(response.code, 200)
        response = self.fetch('/hello', request_timeout=0)
        self.assertEqual(response.code, 200)
        response = self.fetch('/hello', connect_timeout=0, request_timeout=0)
        self.assertEqual(response.code, 200)

    @gen_test
    def test_error_after_cancel(self):
        fut = self.http_client.fetch(self.get_url('/404'))
        self.assertTrue(fut.cancel())
        with ExpectLog(app_log, 'Exception after Future was cancelled') as el:
            for i in range(100):
                yield gen.sleep(0.01)
                if el.logged_stack:
                    break