from dummyserver.server import (
from dummyserver.testcase import SocketDummyServerTestCase, consume_socket
from urllib3 import HTTPConnectionPool, HTTPSConnectionPool, ProxyManager, util
from urllib3._collections import HTTPHeaderDict
from urllib3.connection import HTTPConnection, _get_default_user_agent
from urllib3.exceptions import (
from urllib3.packages.six.moves import http_client as httplib
from urllib3.poolmanager import proxy_from_url
from urllib3.util import ssl_, ssl_wrap_socket
from urllib3.util.retry import Retry
from urllib3.util.timeout import Timeout
from .. import LogRecorder, has_alpn, onlyPy3
import os
import os.path
import select
import shutil
import socket
import ssl
import sys
import tempfile
from collections import OrderedDict
from test import (
from threading import Event
import mock
import pytest
import trustme
class TestProxyManager(SocketDummyServerTestCase):

    def test_simple(self):

        def echo_socket_handler(listener):
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n%s' % (len(buf), buf.decode('utf-8'))).encode('utf-8'))
            sock.close()
        self._start_server(echo_socket_handler)
        base_url = 'http://%s:%d' % (self.host, self.port)
        with proxy_from_url(base_url) as proxy:
            r = proxy.request('GET', 'http://google.com/')
            assert r.status == 200
            assert sorted(r.data.split(b'\r\n')) == sorted([b'GET http://google.com/ HTTP/1.1', b'Host: google.com', b'Accept-Encoding: identity', b'Accept: */*', b'User-Agent: ' + _get_default_user_agent().encode('utf-8'), b'', b''])

    def test_headers(self):

        def echo_socket_handler(listener):
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n%s' % (len(buf), buf.decode('utf-8'))).encode('utf-8'))
            sock.close()
        self._start_server(echo_socket_handler)
        base_url = 'http://%s:%d' % (self.host, self.port)
        proxy_headers = HTTPHeaderDict({'For The Proxy': 'YEAH!'})
        with proxy_from_url(base_url, proxy_headers=proxy_headers) as proxy:
            conn = proxy.connection_from_url('http://www.google.com/')
            r = conn.urlopen('GET', 'http://www.google.com/', assert_same_host=False)
            assert r.status == 200
            assert b'For The Proxy: YEAH!\r\n' in r.data

    def test_retries(self):
        close_event = Event()

        def echo_socket_handler(listener):
            sock = listener.accept()[0]
            sock.close()
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n%s' % (len(buf), buf.decode('utf-8'))).encode('utf-8'))
            sock.close()
            close_event.set()
        self._start_server(echo_socket_handler)
        base_url = 'http://%s:%d' % (self.host, self.port)
        with proxy_from_url(base_url) as proxy:
            conn = proxy.connection_from_url('http://www.google.com')
            r = conn.urlopen('GET', 'http://www.google.com', assert_same_host=False, retries=1)
            assert r.status == 200
            close_event.wait(timeout=LONG_TIMEOUT)
            with pytest.raises(ProxyError):
                conn.urlopen('GET', 'http://www.google.com', assert_same_host=False, retries=False)

    def test_connect_reconn(self):

        def proxy_ssl_one(listener):
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            s = buf.decode('utf-8')
            if not s.startswith('CONNECT '):
                sock.send('HTTP/1.1 405 Method not allowed\r\nAllow: CONNECT\r\n\r\n'.encode('utf-8'))
                sock.close()
                return
            if not s.startswith('CONNECT %s:443' % (self.host,)):
                sock.send('HTTP/1.1 403 Forbidden\r\n\r\n'.encode('utf-8'))
                sock.close()
                return
            sock.send('HTTP/1.1 200 Connection Established\r\n\r\n'.encode('utf-8'))
            ssl_sock = ssl.wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'], ca_certs=DEFAULT_CA)
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += ssl_sock.recv(65536)
            ssl_sock.send('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\nHi'.encode('utf-8'))
            ssl_sock.close()

        def echo_socket_handler(listener):
            proxy_ssl_one(listener)
            proxy_ssl_one(listener)
        self._start_server(echo_socket_handler)
        base_url = 'http://%s:%d' % (self.host, self.port)
        with proxy_from_url(base_url, ca_certs=DEFAULT_CA) as proxy:
            url = 'https://{0}'.format(self.host)
            conn = proxy.connection_from_url(url)
            r = conn.urlopen('GET', url, retries=0)
            assert r.status == 200
            r = conn.urlopen('GET', url, retries=0)
            assert r.status == 200

    def test_connect_ipv6_addr(self):
        ipv6_addr = '2001:4998:c:a06::2:4008'

        def echo_socket_handler(listener):
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            s = buf.decode('utf-8')
            if s.startswith('CONNECT [%s]:443' % (ipv6_addr,)):
                sock.send(b'HTTP/1.1 200 Connection Established\r\n\r\n')
                ssl_sock = ssl.wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'])
                buf = b''
                while not buf.endswith(b'\r\n\r\n'):
                    buf += ssl_sock.recv(65536)
                ssl_sock.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\nHi')
                ssl_sock.close()
            else:
                sock.close()
        self._start_server(echo_socket_handler)
        base_url = 'http://%s:%d' % (self.host, self.port)
        with proxy_from_url(base_url, cert_reqs='NONE') as proxy:
            url = 'https://[{0}]'.format(ipv6_addr)
            conn = proxy.connection_from_url(url)
            try:
                r = conn.urlopen('GET', url, retries=0)
                assert r.status == 200
            except MaxRetryError:
                self.fail('Invalid IPv6 format in HTTP CONNECT request')

    @pytest.mark.parametrize('target_scheme', ['http', 'https'])
    def test_https_proxymanager_connected_to_http_proxy(self, target_scheme):
        if target_scheme == 'https' and sys.version_info[0] == 2:
            pytest.skip("HTTPS-in-HTTPS isn't supported on Python 2")
        errored = Event()

        def http_socket_handler(listener):
            sock = listener.accept()[0]
            sock.send(b'HTTP/1.0 501 Not Implemented\r\nConnection: close\r\n\r\n')
            errored.wait()
            sock.close()
        self._start_server(http_socket_handler)
        base_url = 'https://%s:%d' % (self.host, self.port)
        with ProxyManager(base_url, cert_reqs='NONE') as proxy:
            with pytest.raises(MaxRetryError) as e:
                proxy.request('GET', '%s://example.com' % target_scheme, retries=0)
            errored.set()
            assert type(e.value.reason) == ProxyError
            assert 'Your proxy appears to only use HTTP and not HTTPS' in str(e.value.reason)