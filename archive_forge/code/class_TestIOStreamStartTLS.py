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
class TestIOStreamStartTLS(AsyncTestCase):

    def setUp(self):
        try:
            super().setUp()
            self.listener, self.port = bind_unused_port()
            self.server_stream = None
            self.server_accepted = Future()
            netutil.add_accept_handler(self.listener, self.accept)
            self.client_stream = IOStream(socket.socket())
            self.io_loop.add_future(self.client_stream.connect(('127.0.0.1', self.port)), self.stop)
            self.wait()
            self.io_loop.add_future(self.server_accepted, self.stop)
            self.wait()
        except Exception as e:
            print(e)
            raise

    def tearDown(self):
        if self.server_stream is not None:
            self.server_stream.close()
        if self.client_stream is not None:
            self.client_stream.close()
        self.io_loop.remove_handler(self.listener.fileno())
        self.listener.close()
        super().tearDown()

    def accept(self, connection, address):
        if self.server_stream is not None:
            self.fail('should only get one connection')
        self.server_stream = IOStream(connection)
        self.server_accepted.set_result(None)

    @gen.coroutine
    def client_send_line(self, line):
        assert self.client_stream is not None
        self.client_stream.write(line)
        assert self.server_stream is not None
        recv_line = (yield self.server_stream.read_until(b'\r\n'))
        self.assertEqual(line, recv_line)

    @gen.coroutine
    def server_send_line(self, line):
        assert self.server_stream is not None
        self.server_stream.write(line)
        assert self.client_stream is not None
        recv_line = (yield self.client_stream.read_until(b'\r\n'))
        self.assertEqual(line, recv_line)

    def client_start_tls(self, ssl_options=None, server_hostname=None):
        assert self.client_stream is not None
        client_stream = self.client_stream
        self.client_stream = None
        return client_stream.start_tls(False, ssl_options, server_hostname)

    def server_start_tls(self, ssl_options=None):
        assert self.server_stream is not None
        server_stream = self.server_stream
        self.server_stream = None
        return server_stream.start_tls(True, ssl_options)

    @gen_test
    def test_start_tls_smtp(self):
        yield self.server_send_line(b'220 mail.example.com ready\r\n')
        yield self.client_send_line(b'EHLO mail.example.com\r\n')
        yield self.server_send_line(b'250-mail.example.com welcome\r\n')
        yield self.server_send_line(b'250 STARTTLS\r\n')
        yield self.client_send_line(b'STARTTLS\r\n')
        yield self.server_send_line(b'220 Go ahead\r\n')
        client_future = self.client_start_tls(dict(cert_reqs=ssl.CERT_NONE))
        server_future = self.server_start_tls(_server_ssl_options())
        self.client_stream = (yield client_future)
        self.server_stream = (yield server_future)
        self.assertTrue(isinstance(self.client_stream, SSLIOStream))
        self.assertTrue(isinstance(self.server_stream, SSLIOStream))
        yield self.client_send_line(b'EHLO mail.example.com\r\n')
        yield self.server_send_line(b'250 mail.example.com welcome\r\n')

    @gen_test
    def test_handshake_fail(self):
        server_future = self.server_start_tls(_server_ssl_options())
        with ExpectLog(gen_log, 'SSL Error'):
            client_future = self.client_start_tls(server_hostname='localhost')
            with self.assertRaises(ssl.SSLError):
                yield client_future
            with self.assertRaises((ssl.SSLError, socket.error)):
                yield server_future

    @gen_test
    def test_check_hostname(self):
        server_future = self.server_start_tls(_server_ssl_options())
        with ExpectLog(gen_log, 'SSL Error'):
            client_future = self.client_start_tls(ssl.create_default_context(), server_hostname='127.0.0.1')
            with self.assertRaises(ssl.SSLError):
                yield client_future
            with self.assertRaises(Exception):
                yield server_future

    @gen_test
    def test_typed_memoryview(self):
        buf = memoryview(bytes(80)).cast('L')
        assert self.server_stream is not None
        yield self.server_stream.write(buf)
        assert self.client_stream is not None
        recv = (yield self.client_stream.read_bytes(buf.nbytes))
        self.assertEqual(bytes(recv), bytes(buf))