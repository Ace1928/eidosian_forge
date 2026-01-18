import os
import platform
import socket
import tempfile
import testtools
from unittest import mock
import eventlet
import eventlet.wsgi
import requests
import webob
from oslo_config import cfg
from oslo_service import sslutils
from oslo_service.tests import base
from oslo_service import wsgi
from oslo_utils import netutils
class TestWSGIServer(WsgiTestCase):
    """WSGI server tests."""

    def setUp(self):
        super(TestWSGIServer, self).setUp()

    def test_no_app(self):
        server = wsgi.Server(self.conf, 'test_app', None)
        self.assertEqual('test_app', server.name)

    def test_custom_max_header_line(self):
        self.config(max_header_line=4096)
        wsgi.Server(self.conf, 'test_custom_max_header_line', None)
        self.assertEqual(eventlet.wsgi.MAX_HEADER_LINE, self.conf.max_header_line)

    def test_start_random_port(self):
        server = wsgi.Server(self.conf, 'test_random_port', None, host='127.0.0.1', port=0)
        server.start()
        self.assertNotEqual(0, server.port)
        server.stop()
        server.wait()

    @testtools.skipIf(not netutils.is_ipv6_enabled(), 'no ipv6 support')
    def test_start_random_port_with_ipv6(self):
        server = wsgi.Server(self.conf, 'test_random_port', None, host='::1', port=0)
        server.start()
        self.assertEqual('::1', server.host)
        self.assertNotEqual(0, server.port)
        server.stop()
        server.wait()

    @testtools.skipIf(platform.mac_ver()[0] != '', 'SO_REUSEADDR behaves differently on OSX, see bug 1436895')
    def test_socket_options_for_simple_server(self):
        self.config(tcp_keepidle=500)
        server = wsgi.Server(self.conf, 'test_socket_options', None, host='127.0.0.1', port=0)
        server.start()
        sock = server.socket
        self.assertEqual(1, sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR))
        self.assertEqual(1, sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE))
        if hasattr(socket, 'TCP_KEEPIDLE'):
            self.assertEqual(self.conf.tcp_keepidle, sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE))
        self.assertFalse(server._server.dead)
        server.stop()
        server.wait()
        self.assertTrue(server._server.dead)

    @testtools.skipIf(not hasattr(socket, 'AF_UNIX'), 'UNIX sockets not supported')
    def test_server_with_unix_socket(self):
        socket_file = self.get_temp_file_path('sock')
        socket_mode = 420
        server = wsgi.Server(self.conf, 'test_socket_options', None, socket_family=socket.AF_UNIX, socket_mode=socket_mode, socket_file=socket_file)
        self.assertEqual(socket_file, server.socket.getsockname())
        self.assertEqual(socket_mode, os.stat(socket_file).st_mode & 511)
        server.start()
        self.assertFalse(server._server.dead)
        server.stop()
        server.wait()
        self.assertTrue(server._server.dead)

    def test_server_pool_waitall(self):
        server = wsgi.Server(self.conf, 'test_server', None, host='127.0.0.1')
        server.start()
        with mock.patch.object(server._pool, 'waitall') as mock_waitall:
            server.stop()
            server.wait()
            mock_waitall.assert_called_once_with()

    def test_uri_length_limit(self):
        eventlet.monkey_patch(os=False, thread=False)
        server = wsgi.Server(self.conf, 'test_uri_length_limit', None, host='127.0.0.1', max_url_len=16384, port=33337)
        server.start()
        self.assertFalse(server._server.dead)
        uri = 'http://127.0.0.1:%d/%s' % (server.port, 10000 * 'x')
        resp = requests.get(uri, proxies={'http': ''})
        eventlet.sleep(0)
        self.assertNotEqual(requests.codes.REQUEST_URI_TOO_LARGE, resp.status_code)
        uri = 'http://127.0.0.1:%d/%s' % (server.port, 20000 * 'x')
        resp = requests.get(uri, proxies={'http': ''})
        eventlet.sleep(0)
        self.assertEqual(requests.codes.REQUEST_URI_TOO_LARGE, resp.status_code)
        server.stop()
        server.wait()

    def test_reset_pool_size_to_default(self):
        server = wsgi.Server(self.conf, 'test_resize', None, host='127.0.0.1', max_url_len=16384)
        server.start()
        server.stop()
        self.assertEqual(0, server._pool.size)
        server.reset()
        server.start()
        self.assertEqual(CONF.wsgi_default_pool_size, server._pool.size)

    def test_client_socket_timeout(self):
        self.config(client_socket_timeout=5)
        with mock.patch.object(eventlet, 'spawn') as mock_spawn:
            server = wsgi.Server(self.conf, 'test_app', None, host='127.0.0.1', port=0)
            server.start()
            _, kwargs = mock_spawn.call_args
            self.assertEqual(self.conf.client_socket_timeout, kwargs['socket_timeout'])
            server.stop()

    def test_wsgi_keep_alive(self):
        self.config(wsgi_keep_alive=False)
        with mock.patch.object(eventlet, 'spawn') as mock_spawn:
            server = wsgi.Server(self.conf, 'test_app', None, host='127.0.0.1', port=0)
            server.start()
            _, kwargs = mock_spawn.call_args
            self.assertEqual(self.conf.wsgi_keep_alive, kwargs['keepalive'])
            server.stop()