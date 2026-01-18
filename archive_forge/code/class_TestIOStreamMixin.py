from tornado.concurrent import Future
from tornado import gen
from tornado import netutil
from tornado.ioloop import IOLoop
from tornado.iostream import (
from tornado.httputil import HTTPHeaders
from tornado.locks import Condition, Event
from tornado.log import gen_log
from tornado.netutil import ssl_options_to_context, ssl_wrap_socket
from tornado.platform.asyncio import AddThreadSelectorEventLoop
from tornado.tcpserver import TCPServer
from tornado.testing import (
from tornado.test.util import (
from tornado.web import RequestHandler, Application
import asyncio
import errno
import hashlib
import logging
import os
import platform
import random
import socket
import ssl
import typing
from unittest import mock
import unittest
class TestIOStreamMixin(TestReadWriteMixin):

    def _make_server_iostream(self, connection, **kwargs):
        raise NotImplementedError()

    def _make_client_iostream(self, connection, **kwargs):
        raise NotImplementedError()

    @gen.coroutine
    def make_iostream_pair(self: typing.Any, **kwargs):
        listener, port = bind_unused_port()
        server_stream_fut = Future()

        def accept_callback(connection, address):
            server_stream_fut.set_result(self._make_server_iostream(connection, **kwargs))
        netutil.add_accept_handler(listener, accept_callback)
        client_stream = self._make_client_iostream(socket.socket(), **kwargs)
        connect_fut = client_stream.connect(('127.0.0.1', port))
        server_stream, client_stream = (yield [server_stream_fut, connect_fut])
        self.io_loop.remove_handler(listener.fileno())
        listener.close()
        raise gen.Return((server_stream, client_stream))

    @gen_test
    def test_connection_refused(self: typing.Any):
        cleanup_func, port = refusing_port()
        self.addCleanup(cleanup_func)
        stream = IOStream(socket.socket())
        stream.set_close_callback(self.stop)
        with ExpectLog(gen_log, '.*', required=False):
            with self.assertRaises(StreamClosedError):
                yield stream.connect(('127.0.0.1', port))
        self.assertTrue(isinstance(stream.error, ConnectionRefusedError), stream.error)

    @gen_test
    def test_gaierror(self: typing.Any):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        stream = IOStream(s)
        stream.set_close_callback(self.stop)
        with mock.patch('socket.socket.connect', side_effect=socket.gaierror(errno.EIO, 'boom')):
            with self.assertRaises(StreamClosedError):
                yield stream.connect(('localhost', 80))
            self.assertTrue(isinstance(stream.error, socket.gaierror))

    @gen_test
    def test_read_until_close_with_error(self: typing.Any):
        server, client = (yield self.make_iostream_pair())
        try:
            with mock.patch('tornado.iostream.BaseIOStream._try_inline_read', side_effect=IOError('boom')):
                with self.assertRaisesRegex(IOError, 'boom'):
                    client.read_until_close()
        finally:
            server.close()
            client.close()

    @skipIfNonUnix
    @skipPypy3V58
    @gen_test
    def test_inline_read_error(self: typing.Any):
        io_loop = IOLoop.current()
        if isinstance(io_loop.selector_loop, AddThreadSelectorEventLoop):
            self.skipTest('AddThreadSelectorEventLoop not supported')
        server, client = (yield self.make_iostream_pair())
        try:
            os.close(server.socket.fileno())
            with self.assertRaises(socket.error):
                server.read_bytes(1)
        finally:
            server.close()
            client.close()

    @skipPypy3V58
    @gen_test
    def test_async_read_error_logging(self):
        server, client = (yield self.make_iostream_pair())
        closed = Event()
        server.set_close_callback(closed.set)
        try:
            server.read_bytes(1)
            client.write(b'a')

            def fake_read_from_fd():
                os.close(server.socket.fileno())
                server.__class__.read_from_fd(server)
            server.read_from_fd = fake_read_from_fd
            with ExpectLog(gen_log, 'error on read'):
                yield closed.wait()
        finally:
            server.close()
            client.close()

    @gen_test
    def test_future_write(self):
        """
        Test that write() Futures are never orphaned.
        """
        m, n = (5000, 1000)
        nproducers = 10
        total_bytes = m * n * nproducers
        server, client = (yield self.make_iostream_pair(max_buffer_size=total_bytes))

        @gen.coroutine
        def produce():
            data = b'x' * m
            for i in range(n):
                yield server.write(data)

        @gen.coroutine
        def consume():
            nread = 0
            while nread < total_bytes:
                res = (yield client.read_bytes(m))
                nread += len(res)
        try:
            yield ([produce() for i in range(nproducers)] + [consume()])
        finally:
            server.close()
            client.close()