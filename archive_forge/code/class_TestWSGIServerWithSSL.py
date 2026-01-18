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
class TestWSGIServerWithSSL(WsgiTestCase):
    """WSGI server with SSL tests."""

    def setUp(self):
        super(TestWSGIServerWithSSL, self).setUp()
        cert_file_name = os.path.join(SSL_CERT_DIR, 'certificate.crt')
        key_file_name = os.path.join(SSL_CERT_DIR, 'privatekey.key')
        eventlet.monkey_patch(os=False, thread=False)
        self.host = '127.0.0.1'
        self.config(cert_file=cert_file_name, key_file=key_file_name, group=sslutils.config_section)

    def test_ssl_server(self):

        def test_app(env, start_response):
            start_response('200 OK', {})
            return ['PONG']
        fake_ssl_server = wsgi.Server(self.conf, 'fake_ssl', test_app, host=self.host, port=0, use_ssl=True)
        fake_ssl_server.start()
        self.assertNotEqual(0, fake_ssl_server.port)
        response = requesting(method='GET', host=self.host, port=fake_ssl_server.port, ca_certs=os.path.join(SSL_CERT_DIR, 'ca.crt'))
        self.assertEqual('PONG', response[-4:])
        fake_ssl_server.stop()
        fake_ssl_server.wait()

    def test_two_servers(self):

        def test_app(env, start_response):
            start_response('200 OK', {})
            return ['PONG']
        fake_ssl_server = wsgi.Server(self.conf, 'fake_ssl', test_app, host='127.0.0.1', port=0, use_ssl=True)
        fake_ssl_server.start()
        self.assertNotEqual(0, fake_ssl_server.port)
        fake_server = wsgi.Server(self.conf, 'fake', test_app, host='127.0.0.1', port=0)
        fake_server.start()
        self.assertNotEqual(0, fake_server.port)
        response = requesting(method='GET', host='127.0.0.1', port=fake_ssl_server.port, ca_certs=os.path.join(SSL_CERT_DIR, 'ca.crt'))
        self.assertEqual('PONG', response[-4:])
        response = requesting(method='GET', host='127.0.0.1', port=fake_server.port)
        self.assertEqual('PONG', response[-4:])
        fake_ssl_server.stop()
        fake_ssl_server.wait()
        fake_server.stop()
        fake_server.wait()

    @testtools.skipIf(platform.mac_ver()[0] != '', 'SO_REUSEADDR behaves differently on OSX, see bug 1436895')
    def test_socket_options_for_ssl_server(self):
        self.config(tcp_keepidle=500)
        server = wsgi.Server(self.conf, 'test_socket_options', None, host='127.0.0.1', port=0, use_ssl=True)
        server.start()
        sock = server.socket
        self.assertEqual(1, sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR))
        self.assertEqual(1, sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE))
        if hasattr(socket, 'TCP_KEEPIDLE'):
            self.assertEqual(CONF.tcp_keepidle, sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE))
        server.stop()
        server.wait()

    def test_app_using_ipv6_and_ssl(self):
        greetings = 'Hello, World!!!'

        @webob.dec.wsgify
        def hello_world(req):
            return greetings
        server = wsgi.Server(self.conf, 'fake_ssl', hello_world, host='::1', port=0, use_ssl=True)
        server.start()
        response = requesting(method='GET', host='::1', port=server.port, ca_certs=os.path.join(SSL_CERT_DIR, 'ca.crt'), address_familly=socket.AF_INET6)
        self.assertEqual(greetings, response[-15:])
        server.stop()
        server.wait()