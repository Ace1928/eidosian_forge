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
class TestIOStreamCheckHostname(AsyncTestCase):

    def setUp(self):
        super().setUp()
        self.listener, self.port = bind_unused_port()

        def accept_callback(connection, address):
            ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_ctx.load_cert_chain(os.path.join(os.path.dirname(__file__), 'test.crt'), os.path.join(os.path.dirname(__file__), 'test.key'))
            connection = ssl_ctx.wrap_socket(connection, server_side=True, do_handshake_on_connect=False)
            SSLIOStream(connection)
        netutil.add_accept_handler(self.listener, accept_callback)
        self.client_ssl_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        self.client_ssl_ctx.load_verify_locations(os.path.join(os.path.dirname(__file__), 'test.crt'))

    def tearDown(self):
        self.io_loop.remove_handler(self.listener.fileno())
        self.listener.close()
        super().tearDown()

    @gen_test
    async def test_match(self):
        stream = SSLIOStream(socket.socket(), ssl_options=self.client_ssl_ctx)
        await stream.connect(('127.0.0.1', self.port), server_hostname='foo.example.com')
        stream.close()

    @gen_test
    async def test_no_match(self):
        stream = SSLIOStream(socket.socket(), ssl_options=self.client_ssl_ctx)
        with ExpectLog(gen_log, '.*alert bad certificate', level=logging.WARNING, required=platform.system() != 'Windows'):
            with self.assertRaises(ssl.SSLCertVerificationError):
                with ExpectLog(gen_log, '.*(certificate verify failed: Hostname mismatch)', level=logging.WARNING):
                    await stream.connect(('127.0.0.1', self.port), server_hostname='bar.example.com')
            if platform.system() != 'Windows':
                await asyncio.sleep(0.1)

    @gen_test
    async def test_check_disabled(self):
        self.client_ssl_ctx.check_hostname = False
        stream = SSLIOStream(socket.socket(), ssl_options=self.client_ssl_ctx)
        await stream.connect(('127.0.0.1', self.port), server_hostname='bar.example.com')