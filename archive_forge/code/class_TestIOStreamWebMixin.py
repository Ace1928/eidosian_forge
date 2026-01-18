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
class TestIOStreamWebMixin(object):

    def _make_client_iostream(self):
        raise NotImplementedError()

    def get_app(self):
        return Application([('/', HelloHandler)])

    def test_connection_closed(self: typing.Any):
        response = self.fetch('/', headers={'Connection': 'close'})
        response.rethrow()

    @gen_test
    def test_read_until_close(self: typing.Any):
        stream = self._make_client_iostream()
        yield stream.connect(('127.0.0.1', self.get_http_port()))
        stream.write(b'GET / HTTP/1.0\r\n\r\n')
        data = (yield stream.read_until_close())
        self.assertTrue(data.startswith(b'HTTP/1.1 200'))
        self.assertTrue(data.endswith(b'Hello'))

    @gen_test
    def test_read_zero_bytes(self: typing.Any):
        self.stream = self._make_client_iostream()
        yield self.stream.connect(('127.0.0.1', self.get_http_port()))
        self.stream.write(b'GET / HTTP/1.0\r\n\r\n')
        data = (yield self.stream.read_bytes(9))
        self.assertEqual(data, b'HTTP/1.1 ')
        data = (yield self.stream.read_bytes(0))
        self.assertEqual(data, b'')
        data = (yield self.stream.read_bytes(3))
        self.assertEqual(data, b'200')
        self.stream.close()

    @gen_test
    def test_write_while_connecting(self: typing.Any):
        stream = self._make_client_iostream()
        connect_fut = stream.connect(('127.0.0.1', self.get_http_port()))
        write_fut = stream.write(b'GET / HTTP/1.0\r\nConnection: close\r\n\r\n')
        self.assertFalse(connect_fut.done())
        it = gen.WaitIterator(connect_fut, write_fut)
        resolved_order = []
        while not it.done():
            yield it.next()
            resolved_order.append(it.current_future)
        self.assertEqual(resolved_order, [connect_fut, write_fut])
        data = (yield stream.read_until_close())
        self.assertTrue(data.endswith(b'Hello'))
        stream.close()

    @gen_test
    def test_future_interface(self: typing.Any):
        """Basic test of IOStream's ability to return Futures."""
        stream = self._make_client_iostream()
        connect_result = (yield stream.connect(('127.0.0.1', self.get_http_port())))
        self.assertIs(connect_result, stream)
        yield stream.write(b'GET / HTTP/1.0\r\n\r\n')
        first_line = (yield stream.read_until(b'\r\n'))
        self.assertEqual(first_line, b'HTTP/1.1 200 OK\r\n')
        header_data = (yield stream.read_until(b'\r\n\r\n'))
        headers = HTTPHeaders.parse(header_data.decode('latin1'))
        content_length = int(headers['Content-Length'])
        body = (yield stream.read_bytes(content_length))
        self.assertEqual(body, b'Hello')
        stream.close()

    @gen_test
    def test_future_close_while_reading(self: typing.Any):
        stream = self._make_client_iostream()
        yield stream.connect(('127.0.0.1', self.get_http_port()))
        yield stream.write(b'GET / HTTP/1.0\r\n\r\n')
        with self.assertRaises(StreamClosedError):
            yield stream.read_bytes(1024 * 1024)
        stream.close()

    @gen_test
    def test_future_read_until_close(self: typing.Any):
        stream = self._make_client_iostream()
        yield stream.connect(('127.0.0.1', self.get_http_port()))
        yield stream.write(b'GET / HTTP/1.0\r\nConnection: close\r\n\r\n')
        yield stream.read_until(b'\r\n\r\n')
        body = (yield stream.read_until_close())
        self.assertEqual(body, b'Hello')
        with self.assertRaises(StreamClosedError):
            stream.read_bytes(1)