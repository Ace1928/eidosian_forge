from concurrent import futures
import logging
import re
import socket
import typing
import unittest
from tornado.concurrent import (
from tornado.escape import utf8, to_unicode
from tornado import gen
from tornado.iostream import IOStream
from tornado.tcpserver import TCPServer
from tornado.testing import AsyncTestCase, bind_unused_port, gen_test
class ClientTestMixin(object):
    client_class = None

    def setUp(self):
        super().setUp()
        self.server = CapServer()
        sock, port = bind_unused_port()
        self.server.add_sockets([sock])
        self.client = self.client_class(port=port)

    def tearDown(self):
        self.server.stop()
        super().tearDown()

    def test_future(self: typing.Any):
        future = self.client.capitalize('hello')
        self.io_loop.add_future(future, self.stop)
        self.wait()
        self.assertEqual(future.result(), 'HELLO')

    def test_future_error(self: typing.Any):
        future = self.client.capitalize('HELLO')
        self.io_loop.add_future(future, self.stop)
        self.wait()
        self.assertRaisesRegex(CapError, 'already capitalized', future.result)

    def test_generator(self: typing.Any):

        @gen.coroutine
        def f():
            result = (yield self.client.capitalize('hello'))
            self.assertEqual(result, 'HELLO')
        self.io_loop.run_sync(f)

    def test_generator_error(self: typing.Any):

        @gen.coroutine
        def f():
            with self.assertRaisesRegex(CapError, 'already capitalized'):
                yield self.client.capitalize('HELLO')
        self.io_loop.run_sync(f)