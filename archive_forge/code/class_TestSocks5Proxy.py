import socket
import threading
from test import SHORT_TIMEOUT
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import IPV4SocketDummyServerTestCase
from urllib3.contrib import socks
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError
class TestSocks5Proxy(IPV4SocketDummyServerTestCase):
    """
    Test the SOCKS proxy in SOCKS5 mode.
    """

    def test_basic_request(self):

        def request_handler(listener):
            sock = listener.accept()[0]
            handler = handle_socks5_negotiation(sock, negotiate=False)
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
        proxy_url = 'socks5://%s:%s' % (self.host, self.port)
        with socks.SOCKSProxyManager(proxy_url) as pm:
            response = pm.request('GET', 'http://16.17.18.19')
            assert response.status == 200
            assert response.data == b''
            assert response.headers['Server'] == 'SocksTestServer'

    def test_local_dns(self):

        def request_handler(listener):
            sock = listener.accept()[0]
            handler = handle_socks5_negotiation(sock, negotiate=False)
            addr, port = next(handler)
            assert addr in ['127.0.0.1', '::1']
            assert port == 80
            handler.send(True)
            while True:
                buf = sock.recv(65535)
                if buf.endswith(b'\r\n\r\n'):
                    break
            sock.sendall(b'HTTP/1.1 200 OK\r\nServer: SocksTestServer\r\nContent-Length: 0\r\n\r\n')
            sock.close()
        self._start_server(request_handler)
        proxy_url = 'socks5://%s:%s' % (self.host, self.port)
        with socks.SOCKSProxyManager(proxy_url) as pm:
            response = pm.request('GET', 'http://localhost')
            assert response.status == 200
            assert response.data == b''
            assert response.headers['Server'] == 'SocksTestServer'

    def test_correct_header_line(self):

        def request_handler(listener):
            sock = listener.accept()[0]
            handler = handle_socks5_negotiation(sock, negotiate=False)
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
        proxy_url = 'socks5h://%s:%s' % (self.host, self.port)
        with socks.SOCKSProxyManager(proxy_url) as pm:
            response = pm.request('GET', 'http://example.com')
            assert response.status == 200

    def test_connection_timeouts(self):
        event = threading.Event()

        def request_handler(listener):
            event.wait()
        self._start_server(request_handler)
        proxy_url = 'socks5h://%s:%s' % (self.host, self.port)
        with socks.SOCKSProxyManager(proxy_url) as pm:
            with pytest.raises(ConnectTimeoutError):
                pm.request('GET', 'http://example.com', timeout=SHORT_TIMEOUT, retries=False)
            event.set()

    def test_connection_failure(self):
        event = threading.Event()

        def request_handler(listener):
            listener.close()
            event.set()
        self._start_server(request_handler)
        proxy_url = 'socks5h://%s:%s' % (self.host, self.port)
        with socks.SOCKSProxyManager(proxy_url) as pm:
            event.wait()
            with pytest.raises(NewConnectionError):
                pm.request('GET', 'http://example.com', retries=False)

    def test_proxy_rejection(self):
        evt = threading.Event()

        def request_handler(listener):
            sock = listener.accept()[0]
            handler = handle_socks5_negotiation(sock, negotiate=False)
            addr, port = next(handler)
            handler.send(False)
            evt.wait()
            sock.close()
        self._start_server(request_handler)
        proxy_url = 'socks5h://%s:%s' % (self.host, self.port)
        with socks.SOCKSProxyManager(proxy_url) as pm:
            with pytest.raises(NewConnectionError):
                pm.request('GET', 'http://example.com', retries=False)
            evt.set()

    def test_socks_with_password(self):

        def request_handler(listener):
            sock = listener.accept()[0]
            handler = handle_socks5_negotiation(sock, negotiate=True, username=b'user', password=b'pass')
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
        proxy_url = 'socks5://%s:%s' % (self.host, self.port)
        with socks.SOCKSProxyManager(proxy_url, username='user', password='pass') as pm:
            response = pm.request('GET', 'http://16.17.18.19')
            assert response.status == 200
            assert response.data == b''
            assert response.headers['Server'] == 'SocksTestServer'

    def test_socks_with_auth_in_url(self):
        """
        Test when we have auth info in url, i.e.
        socks5://user:pass@host:port and no username/password as params
        """

        def request_handler(listener):
            sock = listener.accept()[0]
            handler = handle_socks5_negotiation(sock, negotiate=True, username=b'user', password=b'pass')
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
        proxy_url = 'socks5://user:pass@%s:%s' % (self.host, self.port)
        with socks.SOCKSProxyManager(proxy_url) as pm:
            response = pm.request('GET', 'http://16.17.18.19')
            assert response.status == 200
            assert response.data == b''
            assert response.headers['Server'] == 'SocksTestServer'

    def test_socks_with_invalid_password(self):

        def request_handler(listener):
            sock = listener.accept()[0]
            handler = handle_socks5_negotiation(sock, negotiate=True, username=b'user', password=b'pass')
            next(handler)
        self._start_server(request_handler)
        proxy_url = 'socks5h://%s:%s' % (self.host, self.port)
        with socks.SOCKSProxyManager(proxy_url, username='user', password='badpass') as pm:
            with pytest.raises(NewConnectionError) as e:
                pm.request('GET', 'http://example.com', retries=False)
            assert 'SOCKS5 authentication failed' in str(e.value)

    def test_source_address_works(self):
        expected_port = _get_free_port(self.host)

        def request_handler(listener):
            sock = listener.accept()[0]
            assert sock.getpeername()[0] == '127.0.0.1'
            assert sock.getpeername()[1] == expected_port
            handler = handle_socks5_negotiation(sock, negotiate=False)
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
        proxy_url = 'socks5://%s:%s' % (self.host, self.port)
        with socks.SOCKSProxyManager(proxy_url, source_address=('127.0.0.1', expected_port)) as pm:
            response = pm.request('GET', 'http://16.17.18.19')
            assert response.status == 200