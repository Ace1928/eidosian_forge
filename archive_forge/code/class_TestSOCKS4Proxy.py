import socket
import threading
from test import SHORT_TIMEOUT
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import IPV4SocketDummyServerTestCase
from urllib3.contrib import socks
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError
class TestSOCKS4Proxy(IPV4SocketDummyServerTestCase):
    """
    Test the SOCKS proxy in SOCKS4 mode.

    Has relatively fewer tests than the SOCKS5 case, mostly because once the
    negotiation is done the two cases behave identically.
    """

    def test_basic_request(self):

        def request_handler(listener):
            sock = listener.accept()[0]
            handler = handle_socks4_negotiation(sock)
            addr, port = next(handler)
            assert addr == '16.17.18.19'
            assert port == 80
            handler.send(True)
            while True:
                buf = sock.recv(65535)
                if buf.endswith(b'\r\n\r\n'):
                    break
            sock.sendall(b'HTTP/1.1 200 OK\r\nServer: SocksTestServer\r\nContent-Length: 0\r\n\r\n')
            sock.close()
        self._start_server(request_handler)
        proxy_url = 'socks4://%s:%s' % (self.host, self.port)
        with socks.SOCKSProxyManager(proxy_url) as pm:
            response = pm.request('GET', 'http://16.17.18.19')
            assert response.status == 200
            assert response.headers['Server'] == 'SocksTestServer'
            assert response.data == b''

    def test_local_dns(self):

        def request_handler(listener):
            sock = listener.accept()[0]
            handler = handle_socks4_negotiation(sock)
            addr, port = next(handler)
            assert addr == '127.0.0.1'
            assert port == 80
            handler.send(True)
            while True:
                buf = sock.recv(65535)
                if buf.endswith(b'\r\n\r\n'):
                    break
            sock.sendall(b'HTTP/1.1 200 OK\r\nServer: SocksTestServer\r\nContent-Length: 0\r\n\r\n')
            sock.close()
        self._start_server(request_handler)
        proxy_url = 'socks4://%s:%s' % (self.host, self.port)
        with socks.SOCKSProxyManager(proxy_url) as pm:
            response = pm.request('GET', 'http://localhost')
            assert response.status == 200
            assert response.headers['Server'] == 'SocksTestServer'
            assert response.data == b''

    def test_correct_header_line(self):

        def request_handler(listener):
            sock = listener.accept()[0]
            handler = handle_socks4_negotiation(sock)
            addr, port = next(handler)
            assert addr == b'example.com'
            assert port == 80
            handler.send(True)
            buf = b''
            while True:
                buf += sock.recv(65535)
                if buf.endswith(b'\r\n\r\n'):
                    break
            assert buf.startswith(b'GET / HTTP/1.1')
            assert b'Host: example.com' in buf
            sock.sendall(b'HTTP/1.1 200 OK\r\nServer: SocksTestServer\r\nContent-Length: 0\r\n\r\n')
            sock.close()
        self._start_server(request_handler)
        proxy_url = 'socks4a://%s:%s' % (self.host, self.port)
        with socks.SOCKSProxyManager(proxy_url) as pm:
            response = pm.request('GET', 'http://example.com')
            assert response.status == 200

    def test_proxy_rejection(self):
        evt = threading.Event()

        def request_handler(listener):
            sock = listener.accept()[0]
            handler = handle_socks4_negotiation(sock)
            addr, port = next(handler)
            handler.send(False)
            evt.wait()
            sock.close()
        self._start_server(request_handler)
        proxy_url = 'socks4a://%s:%s' % (self.host, self.port)
        with socks.SOCKSProxyManager(proxy_url) as pm:
            with pytest.raises(NewConnectionError):
                pm.request('GET', 'http://example.com', retries=False)
            evt.set()

    def test_socks4_with_username(self):

        def request_handler(listener):
            sock = listener.accept()[0]
            handler = handle_socks4_negotiation(sock, username=b'user')
            addr, port = next(handler)
            assert addr == '16.17.18.19'
            assert port == 80
            handler.send(True)
            while True:
                buf = sock.recv(65535)
                if buf.endswith(b'\r\n\r\n'):
                    break
            sock.sendall(b'HTTP/1.1 200 OK\r\nServer: SocksTestServer\r\nContent-Length: 0\r\n\r\n')
            sock.close()
        self._start_server(request_handler)
        proxy_url = 'socks4://%s:%s' % (self.host, self.port)
        with socks.SOCKSProxyManager(proxy_url, username='user') as pm:
            response = pm.request('GET', 'http://16.17.18.19')
            assert response.status == 200
            assert response.data == b''
            assert response.headers['Server'] == 'SocksTestServer'

    def test_socks_with_invalid_username(self):

        def request_handler(listener):
            sock = listener.accept()[0]
            handler = handle_socks4_negotiation(sock, username=b'user')
            next(handler)
        self._start_server(request_handler)
        proxy_url = 'socks4a://%s:%s' % (self.host, self.port)
        with socks.SOCKSProxyManager(proxy_url, username='baduser') as pm:
            with pytest.raises(NewConnectionError) as e:
                pm.request('GET', 'http://example.com', retries=False)
                assert 'different user-ids' in str(e.value)