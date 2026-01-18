from contextlib import closing
import getpass
import os
import socket
import unittest
from tornado.concurrent import Future
from tornado.netutil import bind_sockets, Resolver
from tornado.queues import Queue
from tornado.tcpclient import TCPClient, _Connector
from tornado.tcpserver import TCPServer
from tornado.testing import AsyncTestCase, gen_test
from tornado.test.util import skipIfNoIPv6, refusing_port, skipIfNonUnix
from tornado.gen import TimeoutError
import typing
class TCPClientTest(AsyncTestCase):

    def setUp(self):
        super().setUp()
        self.server = None
        self.client = TCPClient()

    def start_server(self, family):
        if family == socket.AF_UNSPEC and 'TRAVIS' in os.environ:
            self.skipTest('dual-stack servers often have port conflicts on travis')
        self.server = TestTCPServer(family)
        return self.server.port

    def stop_server(self):
        if self.server is not None:
            self.server.stop()
            self.server = None

    def tearDown(self):
        self.client.close()
        self.stop_server()
        super().tearDown()

    def skipIfLocalhostV4(self):
        addrinfo = self.io_loop.run_sync(lambda: Resolver().resolve('localhost', 80))
        families = set((addr[0] for addr in addrinfo))
        if socket.AF_INET6 not in families:
            self.skipTest('localhost does not resolve to ipv6')

    @gen_test
    def do_test_connect(self, family, host, source_ip=None, source_port=None):
        port = self.start_server(family)
        stream = (yield self.client.connect(host, port, source_ip=source_ip, source_port=source_port, af=family))
        assert self.server is not None
        server_stream = (yield self.server.queue.get())
        with closing(stream):
            stream.write(b'hello')
            data = (yield server_stream.read_bytes(5))
            self.assertEqual(data, b'hello')

    def test_connect_ipv4_ipv4(self):
        self.do_test_connect(socket.AF_INET, '127.0.0.1')

    def test_connect_ipv4_dual(self):
        self.do_test_connect(socket.AF_INET, 'localhost')

    @skipIfNoIPv6
    def test_connect_ipv6_ipv6(self):
        self.skipIfLocalhostV4()
        self.do_test_connect(socket.AF_INET6, '::1')

    @skipIfNoIPv6
    def test_connect_ipv6_dual(self):
        self.skipIfLocalhostV4()
        if Resolver.configured_class().__name__.endswith('TwistedResolver'):
            self.skipTest('TwistedResolver does not support multiple addresses')
        self.do_test_connect(socket.AF_INET6, 'localhost')

    def test_connect_unspec_ipv4(self):
        self.do_test_connect(socket.AF_UNSPEC, '127.0.0.1')

    @skipIfNoIPv6
    def test_connect_unspec_ipv6(self):
        self.skipIfLocalhostV4()
        self.do_test_connect(socket.AF_UNSPEC, '::1')

    def test_connect_unspec_dual(self):
        self.do_test_connect(socket.AF_UNSPEC, 'localhost')

    @gen_test
    def test_refused_ipv4(self):
        cleanup_func, port = refusing_port()
        self.addCleanup(cleanup_func)
        with self.assertRaises(IOError):
            yield self.client.connect('127.0.0.1', port)

    def test_source_ip_fail(self):
        """Fail when trying to use the source IP Address '8.8.8.8'."""
        self.assertRaises(socket.error, self.do_test_connect, socket.AF_INET, '127.0.0.1', source_ip='8.8.8.8')

    def test_source_ip_success(self):
        """Success when trying to use the source IP Address '127.0.0.1'."""
        self.do_test_connect(socket.AF_INET, '127.0.0.1', source_ip='127.0.0.1')

    @skipIfNonUnix
    def test_source_port_fail(self):
        """Fail when trying to use source port 1."""
        if getpass.getuser() == 'root':
            self.skipTest('running as root')
        self.assertRaises(socket.error, self.do_test_connect, socket.AF_INET, '127.0.0.1', source_port=1)

    @gen_test
    def test_connect_timeout(self):
        timeout = 0.05

        class TimeoutResolver(Resolver):

            def resolve(self, *args, **kwargs):
                return Future()
        with self.assertRaises(TimeoutError):
            yield TCPClient(resolver=TimeoutResolver()).connect('1.2.3.4', 12345, timeout=timeout)