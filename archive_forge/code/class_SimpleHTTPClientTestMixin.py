import collections
from contextlib import closing
import errno
import logging
import os
import re
import socket
import ssl
import sys
import typing  # noqa: F401
from tornado.escape import to_unicode, utf8
from tornado import gen, version
from tornado.httpclient import AsyncHTTPClient
from tornado.httputil import HTTPHeaders, ResponseStartLine
from tornado.ioloop import IOLoop
from tornado.iostream import UnsatisfiableReadError
from tornado.locks import Event
from tornado.log import gen_log
from tornado.netutil import Resolver, bind_sockets
from tornado.simple_httpclient import (
from tornado.test.httpclient_test import (
from tornado.test import httpclient_test
from tornado.testing import (
from tornado.test.util import skipOnTravis, skipIfNoIPv6, refusing_port
from tornado.web import RequestHandler, Application, url, stream_request_body
class SimpleHTTPClientTestMixin(object):

    def create_client(self, **kwargs):
        raise NotImplementedError()

    def get_app(self: typing.Any):
        self.triggers = collections.deque()
        return Application([url('/trigger', TriggerHandler, dict(queue=self.triggers, wake_callback=self.stop)), url('/chunk', ChunkHandler), url('/countdown/([0-9]+)', CountdownHandler, name='countdown'), url('/hello', HelloWorldHandler), url('/content_length', ContentLengthHandler), url('/head', HeadHandler), url('/options', OptionsHandler), url('/no_content', NoContentHandler), url('/see_other_post', SeeOtherPostHandler), url('/see_other_get', SeeOtherGetHandler), url('/host_echo', HostEchoHandler), url('/no_content_length', NoContentLengthHandler), url('/echo_post', EchoPostHandler), url('/respond_in_prepare', RespondInPrepareHandler), url('/redirect', RedirectHandler), url('/user_agent', UserAgentHandler)], gzip=True)

    def test_singleton(self: typing.Any):
        self.assertTrue(SimpleAsyncHTTPClient() is SimpleAsyncHTTPClient())
        self.assertTrue(SimpleAsyncHTTPClient() is not SimpleAsyncHTTPClient(force_instance=True))
        with closing(IOLoop()) as io_loop2:

            async def make_client():
                await gen.sleep(0)
                return SimpleAsyncHTTPClient()
            client1 = self.io_loop.run_sync(make_client)
            client2 = io_loop2.run_sync(make_client)
            self.assertTrue(client1 is not client2)

    def test_connection_limit(self: typing.Any):
        with closing(self.create_client(max_clients=2)) as client:
            self.assertEqual(client.max_clients, 2)
            seen = []
            for i in range(4):

                def cb(fut, i=i):
                    seen.append(i)
                    self.stop()
                client.fetch(self.get_url('/trigger')).add_done_callback(cb)
            self.wait(condition=lambda: len(self.triggers) == 2)
            self.assertEqual(len(client.queue), 2)
            self.triggers.popleft()()
            self.triggers.popleft()()
            self.wait(condition=lambda: len(self.triggers) == 2 and len(seen) == 2)
            self.assertEqual(set(seen), set([0, 1]))
            self.assertEqual(len(client.queue), 0)
            self.triggers.popleft()()
            self.triggers.popleft()()
            self.wait(condition=lambda: len(seen) == 4)
            self.assertEqual(set(seen), set([0, 1, 2, 3]))
            self.assertEqual(len(self.triggers), 0)

    @gen_test
    def test_redirect_connection_limit(self: typing.Any):
        with closing(self.create_client(max_clients=1)) as client:
            response = (yield client.fetch(self.get_url('/countdown/3'), max_redirects=3))
            response.rethrow()

    def test_max_redirects(self: typing.Any):
        response = self.fetch('/countdown/5', max_redirects=3)
        self.assertEqual(302, response.code)
        self.assertTrue(response.request.url.endswith('/countdown/5'))
        self.assertTrue(response.effective_url.endswith('/countdown/2'))
        self.assertTrue(response.headers['Location'].endswith('/countdown/1'))

    def test_header_reuse(self: typing.Any):
        headers = HTTPHeaders({'User-Agent': 'Foo'})
        self.fetch('/hello', headers=headers)
        self.assertEqual(list(headers.get_all()), [('User-Agent', 'Foo')])

    def test_default_user_agent(self: typing.Any):
        response = self.fetch('/user_agent', method='GET')
        self.assertEqual(200, response.code)
        self.assertEqual(response.body.decode(), 'Tornado/{}'.format(version))

    def test_see_other_redirect(self: typing.Any):
        for code in (302, 303):
            response = self.fetch('/see_other_post', method='POST', body='%d' % code)
            self.assertEqual(200, response.code)
            self.assertTrue(response.request.url.endswith('/see_other_post'))
            self.assertTrue(response.effective_url.endswith('/see_other_get'))
            self.assertEqual('POST', response.request.method)

    @skipOnTravis
    @gen_test
    def test_connect_timeout(self: typing.Any):
        timeout = 0.1
        cleanup_event = Event()
        test = self

        class TimeoutResolver(Resolver):

            async def resolve(self, *args, **kwargs):
                await cleanup_event.wait()
                return [(socket.AF_INET, ('127.0.0.1', test.get_http_port()))]
        with closing(self.create_client(resolver=TimeoutResolver())) as client:
            with self.assertRaises(HTTPTimeoutError):
                yield client.fetch(self.get_url('/hello'), connect_timeout=timeout, request_timeout=3600, raise_error=True)
        cleanup_event.set()
        yield gen.sleep(0.2)

    @skipOnTravis
    def test_request_timeout(self: typing.Any):
        timeout = 0.1
        if os.name == 'nt':
            timeout = 0.5
        with self.assertRaises(HTTPTimeoutError):
            self.fetch('/trigger?wake=false', request_timeout=timeout, raise_error=True)
        self.triggers.popleft()()
        self.io_loop.run_sync(lambda: gen.sleep(0))

    @skipIfNoIPv6
    def test_ipv6(self: typing.Any):
        [sock] = bind_sockets(0, '::1', family=socket.AF_INET6)
        port = sock.getsockname()[1]
        self.http_server.add_socket(sock)
        url = '%s://[::1]:%d/hello' % (self.get_protocol(), port)
        with self.assertRaises(Exception):
            self.fetch(url, allow_ipv6=False, raise_error=True)
        response = self.fetch(url)
        self.assertEqual(response.body, b'Hello world!')

    def test_multiple_content_length_accepted(self: typing.Any):
        response = self.fetch('/content_length?value=2,2')
        self.assertEqual(response.body, b'ok')
        response = self.fetch('/content_length?value=2,%202,2')
        self.assertEqual(response.body, b'ok')
        with ExpectLog(gen_log, '.*Multiple unequal Content-Lengths', level=logging.INFO):
            with self.assertRaises(HTTPStreamClosedError):
                self.fetch('/content_length?value=2,4', raise_error=True)
            with self.assertRaises(HTTPStreamClosedError):
                self.fetch('/content_length?value=2,%202,3', raise_error=True)

    def test_head_request(self: typing.Any):
        response = self.fetch('/head', method='HEAD')
        self.assertEqual(response.code, 200)
        self.assertEqual(response.headers['content-length'], '7')
        self.assertFalse(response.body)

    def test_options_request(self: typing.Any):
        response = self.fetch('/options', method='OPTIONS')
        self.assertEqual(response.code, 200)
        self.assertEqual(response.headers['content-length'], '2')
        self.assertEqual(response.headers['access-control-allow-origin'], '*')
        self.assertEqual(response.body, b'ok')

    def test_no_content(self: typing.Any):
        response = self.fetch('/no_content')
        self.assertEqual(response.code, 204)
        self.assertNotIn('Content-Length', response.headers)

    def test_host_header(self: typing.Any):
        host_re = re.compile(b'^127.0.0.1:[0-9]+$')
        response = self.fetch('/host_echo')
        self.assertTrue(host_re.match(response.body))
        url = self.get_url('/host_echo').replace('http://', 'http://me:secret@')
        response = self.fetch(url)
        self.assertTrue(host_re.match(response.body), response.body)

    def test_connection_refused(self: typing.Any):
        cleanup_func, port = refusing_port()
        self.addCleanup(cleanup_func)
        with ExpectLog(gen_log, '.*', required=False):
            with self.assertRaises(socket.error) as cm:
                self.fetch('http://127.0.0.1:%d/' % port, raise_error=True)
        if sys.platform != 'cygwin':
            contains_errno = str(errno.ECONNREFUSED) in str(cm.exception)
            if not contains_errno and hasattr(errno, 'WSAECONNREFUSED'):
                contains_errno = str(errno.WSAECONNREFUSED) in str(cm.exception)
            self.assertTrue(contains_errno, cm.exception)
            expected_message = os.strerror(errno.ECONNREFUSED)
            self.assertTrue(expected_message in str(cm.exception), cm.exception)

    def test_queue_timeout(self: typing.Any):
        with closing(self.create_client(max_clients=1)) as client:
            fut1 = client.fetch(self.get_url('/trigger'), request_timeout=10)
            self.wait()
            with self.assertRaises(HTTPTimeoutError) as cm:
                self.io_loop.run_sync(lambda: client.fetch(self.get_url('/hello'), connect_timeout=0.1, raise_error=True))
            self.assertEqual(str(cm.exception), 'Timeout in request queue')
            self.triggers.popleft()()
            self.io_loop.run_sync(lambda: fut1)

    def test_no_content_length(self: typing.Any):
        response = self.fetch('/no_content_length')
        if response.body == b'HTTP/1 required':
            self.skipTest('requires HTTP/1.x')
        else:
            self.assertEqual(b'hello', response.body)

    def sync_body_producer(self, write):
        write(b'1234')
        write(b'5678')

    @gen.coroutine
    def async_body_producer(self, write):
        yield write(b'1234')
        yield gen.moment
        yield write(b'5678')

    def test_sync_body_producer_chunked(self: typing.Any):
        response = self.fetch('/echo_post', method='POST', body_producer=self.sync_body_producer)
        response.rethrow()
        self.assertEqual(response.body, b'12345678')

    def test_sync_body_producer_content_length(self: typing.Any):
        response = self.fetch('/echo_post', method='POST', body_producer=self.sync_body_producer, headers={'Content-Length': '8'})
        response.rethrow()
        self.assertEqual(response.body, b'12345678')

    def test_async_body_producer_chunked(self: typing.Any):
        response = self.fetch('/echo_post', method='POST', body_producer=self.async_body_producer)
        response.rethrow()
        self.assertEqual(response.body, b'12345678')

    def test_async_body_producer_content_length(self: typing.Any):
        response = self.fetch('/echo_post', method='POST', body_producer=self.async_body_producer, headers={'Content-Length': '8'})
        response.rethrow()
        self.assertEqual(response.body, b'12345678')

    def test_native_body_producer_chunked(self: typing.Any):

        async def body_producer(write):
            await write(b'1234')
            import asyncio
            await asyncio.sleep(0)
            await write(b'5678')
        response = self.fetch('/echo_post', method='POST', body_producer=body_producer)
        response.rethrow()
        self.assertEqual(response.body, b'12345678')

    def test_native_body_producer_content_length(self: typing.Any):

        async def body_producer(write):
            await write(b'1234')
            import asyncio
            await asyncio.sleep(0)
            await write(b'5678')
        response = self.fetch('/echo_post', method='POST', body_producer=body_producer, headers={'Content-Length': '8'})
        response.rethrow()
        self.assertEqual(response.body, b'12345678')

    def test_100_continue(self: typing.Any):
        response = self.fetch('/echo_post', method='POST', body=b'1234', expect_100_continue=True)
        self.assertEqual(response.body, b'1234')

    def test_100_continue_early_response(self: typing.Any):

        def body_producer(write):
            raise Exception('should not be called')
        response = self.fetch('/respond_in_prepare', method='POST', body_producer=body_producer, expect_100_continue=True)
        self.assertEqual(response.code, 403)

    def test_streaming_follow_redirects(self: typing.Any):
        headers = []
        chunk_bytes = []
        self.fetch('/redirect?url=/hello', header_callback=headers.append, streaming_callback=chunk_bytes.append)
        chunks = list(map(to_unicode, chunk_bytes))
        self.assertEqual(chunks, ['Hello world!'])
        num_start_lines = len([h for h in headers if h.startswith('HTTP/')])
        self.assertEqual(num_start_lines, 1)