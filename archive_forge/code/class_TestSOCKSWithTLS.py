import socket
import threading
from test import SHORT_TIMEOUT
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import IPV4SocketDummyServerTestCase
from urllib3.contrib import socks
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError
class TestSOCKSWithTLS(IPV4SocketDummyServerTestCase):
    """
    Test that TLS behaves properly for SOCKS proxies.
    """

    @pytest.mark.skipif(not HAS_SSL, reason='No TLS available')
    def test_basic_request(self):

        def request_handler(listener):
            sock = listener.accept()[0]
            handler = handle_socks5_negotiation(sock, negotiate=False)
            addr, port = next(handler)
            assert addr == b'localhost'
            assert port == 443
            handler.send(True)
            context = better_ssl.SSLContext(ssl.PROTOCOL_SSLv23)
            context.load_cert_chain(DEFAULT_CERTS['certfile'], DEFAULT_CERTS['keyfile'])
            tls = context.wrap_socket(sock, server_side=True)
            buf = b''
            while True:
                buf += tls.recv(65535)
                if buf.endswith(b'\r\n\r\n'):
                    break
            assert buf.startswith(b'GET / HTTP/1.1\r\n')
            tls.sendall(b'HTTP/1.1 200 OK\r\nServer: SocksTestServer\r\nContent-Length: 0\r\n\r\n')
            tls.close()
            sock.close()
        self._start_server(request_handler)
        proxy_url = 'socks5h://%s:%s' % (self.host, self.port)
        with socks.SOCKSProxyManager(proxy_url, ca_certs=DEFAULT_CA) as pm:
            response = pm.request('GET', 'https://localhost')
            assert response.status == 200
            assert response.data == b''
            assert response.headers['Server'] == 'SocksTestServer'