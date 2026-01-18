import socket
import subprocess
import sys
import textwrap
import unittest
from tornado import gen
from tornado.iostream import IOStream
from tornado.log import app_log
from tornado.tcpserver import TCPServer
from tornado.test.util import skipIfNonUnix
from tornado.testing import AsyncTestCase, ExpectLog, bind_unused_port, gen_test
from typing import Tuple
class TCPServerTest(AsyncTestCase):

    @gen_test
    def test_handle_stream_coroutine_logging(self):

        class TestServer(TCPServer):

            @gen.coroutine
            def handle_stream(self, stream, address):
                yield stream.read_bytes(len(b'hello'))
                stream.close()
                1 / 0
        server = client = None
        try:
            sock, port = bind_unused_port()
            server = TestServer()
            server.add_socket(sock)
            client = IOStream(socket.socket())
            with ExpectLog(app_log, 'Exception in callback'):
                yield client.connect(('localhost', port))
                yield client.write(b'hello')
                yield client.read_until_close()
                yield gen.moment
        finally:
            if server is not None:
                server.stop()
            if client is not None:
                client.close()

    @gen_test
    def test_handle_stream_native_coroutine(self):

        class TestServer(TCPServer):

            async def handle_stream(self, stream, address):
                stream.write(b'data')
                stream.close()
        sock, port = bind_unused_port()
        server = TestServer()
        server.add_socket(sock)
        client = IOStream(socket.socket())
        yield client.connect(('localhost', port))
        result = (yield client.read_until_close())
        self.assertEqual(result, b'data')
        server.stop()
        client.close()

    def test_stop_twice(self):
        sock, port = bind_unused_port()
        server = TCPServer()
        server.add_socket(sock)
        server.stop()
        server.stop()

    @gen_test
    def test_stop_in_callback(self):

        class TestServer(TCPServer):

            @gen.coroutine
            def handle_stream(self, stream, address):
                server.stop()
                yield stream.read_until_close()
        sock, port = bind_unused_port()
        server = TestServer()
        server.add_socket(sock)
        server_addr = ('localhost', port)
        N = 40
        clients = [IOStream(socket.socket()) for i in range(N)]
        connected_clients = []

        @gen.coroutine
        def connect(c):
            try:
                yield c.connect(server_addr)
            except EnvironmentError:
                pass
            else:
                connected_clients.append(c)
        yield [connect(c) for c in clients]
        self.assertGreater(len(connected_clients), 0, 'all clients failed connecting')
        try:
            if len(connected_clients) == N:
                self.skipTest('at least one client should fail connecting for the test to be meaningful')
        finally:
            for c in connected_clients:
                c.close()