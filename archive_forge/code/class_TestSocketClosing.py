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
class TestSocketClosing(SocketDummyServerTestCase):

    def test_recovery_when_server_closes_connection(self):
        done_closing = Event()

        def socket_handler(listener):
            for i in (0, 1):
                sock = listener.accept()[0]
                buf = b''
                while not buf.endswith(b'\r\n\r\n'):
                    buf = sock.recv(65536)
                body = 'Response %d' % i
                sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n%s' % (len(body), body)).encode('utf-8'))
                sock.close()
                done_closing.set()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            response = pool.request('GET', '/', retries=0)
            assert response.status == 200
            assert response.data == b'Response 0'
            done_closing.wait()
            response = pool.request('GET', '/', retries=0)
            assert response.status == 200
            assert response.data == b'Response 1'

    def test_connection_refused(self):
        host, port = get_unreachable_address()
        with HTTPConnectionPool(host, port, maxsize=3, block=True) as http:
            with pytest.raises(MaxRetryError):
                http.request('GET', '/', retries=0, release_conn=False)
            assert http.pool.qsize() == http.pool.maxsize

    def test_connection_read_timeout(self):
        timed_out = Event()

        def socket_handler(listener):
            sock = listener.accept()[0]
            while not sock.recv(65536).endswith(b'\r\n\r\n'):
                pass
            timed_out.wait()
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port, timeout=SHORT_TIMEOUT, retries=False, maxsize=3, block=True) as http:
            try:
                with pytest.raises(ReadTimeoutError):
                    http.request('GET', '/', release_conn=False)
            finally:
                timed_out.set()
            assert http.pool.qsize() == http.pool.maxsize

    def test_read_timeout_dont_retry_method_not_in_allowlist(self):
        timed_out = Event()

        def socket_handler(listener):
            sock = listener.accept()[0]
            sock.recv(65536)
            timed_out.wait()
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port, timeout=LONG_TIMEOUT, retries=True) as pool:
            try:
                with pytest.raises(ReadTimeoutError):
                    pool.request('POST', '/')
            finally:
                timed_out.set()

    def test_https_connection_read_timeout(self):
        """Handshake timeouts should fail with a Timeout"""
        timed_out = Event()

        def socket_handler(listener):
            sock = listener.accept()[0]
            while not sock.recv(65536):
                pass
            timed_out.wait()
            sock.close()
        self._start_server(socket_handler)
        with HTTPSConnectionPool(self.host, self.port, timeout=LONG_TIMEOUT, retries=False) as pool:
            try:
                with pytest.raises(ReadTimeoutError):
                    pool.request('GET', '/')
            finally:
                timed_out.set()

    def test_timeout_errors_cause_retries(self):

        def socket_handler(listener):
            sock_timeout = listener.accept()[0]
            sock = listener.accept()[0]
            sock_timeout.close()
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            body = 'Response 2'
            sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n%s' % (len(body), body)).encode('utf-8'))
            sock.close()
        default_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(1)
        try:
            self._start_server(socket_handler)
            t = Timeout(connect=LONG_TIMEOUT, read=LONG_TIMEOUT)
            with HTTPConnectionPool(self.host, self.port, timeout=t) as pool:
                response = pool.request('GET', '/', retries=1)
                assert response.status == 200
                assert response.data == b'Response 2'
        finally:
            socket.setdefaulttimeout(default_timeout)

    def test_delayed_body_read_timeout(self):
        timed_out = Event()

        def socket_handler(listener):
            sock = listener.accept()[0]
            buf = b''
            body = 'Hi'
            while not buf.endswith(b'\r\n\r\n'):
                buf = sock.recv(65536)
            sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n' % len(body)).encode('utf-8'))
            timed_out.wait()
            sock.send(body.encode('utf-8'))
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            response = pool.urlopen('GET', '/', retries=0, preload_content=False, timeout=Timeout(connect=1, read=LONG_TIMEOUT))
            try:
                with pytest.raises(ReadTimeoutError):
                    response.read()
            finally:
                timed_out.set()

    def test_delayed_body_read_timeout_with_preload(self):
        timed_out = Event()

        def socket_handler(listener):
            sock = listener.accept()[0]
            buf = b''
            body = 'Hi'
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n' % len(body)).encode('utf-8'))
            timed_out.wait(5)
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            try:
                with pytest.raises(ReadTimeoutError):
                    timeout = Timeout(connect=LONG_TIMEOUT, read=SHORT_TIMEOUT)
                    pool.urlopen('GET', '/', retries=False, timeout=timeout)
            finally:
                timed_out.set()

    def test_incomplete_response(self):
        body = 'Response'
        partial_body = body[:2]

        def socket_handler(listener):
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf = sock.recv(65536)
            sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n%s' % (len(body), partial_body)).encode('utf-8'))
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            response = pool.request('GET', '/', retries=0, preload_content=False)
            with pytest.raises(ProtocolError):
                response.read()

    def test_retry_weird_http_version(self):
        """Retry class should handle httplib.BadStatusLine errors properly"""

        def socket_handler(listener):
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            body = 'bad http 0.5 response'
            sock.send(('HTTP/0.5 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n%s' % (len(body), body)).encode('utf-8'))
            sock.close()
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\nfoo' % len('foo')).encode('utf-8'))
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            retry = Retry(read=1)
            response = pool.request('GET', '/', retries=retry)
            assert response.status == 200
            assert response.data == b'foo'

    def test_connection_cleanup_on_read_timeout(self):
        timed_out = Event()

        def socket_handler(listener):
            sock = listener.accept()[0]
            buf = b''
            body = 'Hi'
            while not buf.endswith(b'\r\n\r\n'):
                buf = sock.recv(65536)
            sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n' % len(body)).encode('utf-8'))
            timed_out.wait()
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            poolsize = pool.pool.qsize()
            response = pool.urlopen('GET', '/', retries=0, preload_content=False, timeout=LONG_TIMEOUT)
            try:
                with pytest.raises(ReadTimeoutError):
                    response.read()
                assert poolsize == pool.pool.qsize()
            finally:
                timed_out.set()

    def test_connection_cleanup_on_protocol_error_during_read(self):
        body = 'Response'
        partial_body = body[:2]

        def socket_handler(listener):
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf = sock.recv(65536)
            sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n%s' % (len(body), partial_body)).encode('utf-8'))
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            poolsize = pool.pool.qsize()
            response = pool.request('GET', '/', retries=0, preload_content=False)
            with pytest.raises(ProtocolError):
                response.read()
            assert poolsize == pool.pool.qsize()

    def test_connection_closed_on_read_timeout_preload_false(self):
        timed_out = Event()

        def socket_handler(listener):
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf = sock.recv(65535)
            sock.send('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nTransfer-Encoding: chunked\r\n\r\n8\r\n12345678\r\n'.encode('utf-8'))
            timed_out.wait(5)
            rlist, _, _ = select.select([listener], [], [], 1)
            assert rlist
            new_sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf = new_sock.recv(65535)
            new_sock.send('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nTransfer-Encoding: chunked\r\n\r\n8\r\n12345678\r\n0\r\n\r\n'.encode('utf-8'))
            new_sock.close()
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            response = pool.urlopen('GET', '/', retries=0, preload_content=False, timeout=LONG_TIMEOUT)
            try:
                with pytest.raises(ReadTimeoutError):
                    response.read()
            finally:
                timed_out.set()
            response = pool.urlopen('GET', '/', retries=0, preload_content=False, timeout=LONG_TIMEOUT)
            assert len(response.read()) == 8

    def test_closing_response_actually_closes_connection(self):
        done_closing = Event()
        complete = Event()

        def socket_handler(listener):
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf = sock.recv(65536)
            sock.send('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 0\r\n\r\n'.encode('utf-8'))
            done_closing.wait(timeout=LONG_TIMEOUT)
            sock.settimeout(LONG_TIMEOUT)
            new_data = sock.recv(65536)
            assert not new_data
            sock.close()
            complete.set()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            response = pool.request('GET', '/', retries=0, preload_content=False)
            assert response.status == 200
            response.close()
            done_closing.set()
            successful = complete.wait(timeout=LONG_TIMEOUT)
            assert successful, 'Timed out waiting for connection close'

    def test_release_conn_param_is_respected_after_timeout_retry(self):
        """For successful ```urlopen(release_conn=False)```,
        the connection isn't released, even after a retry.

        This test allows a retry: one request fails, the next request succeeds.

        This is a regression test for issue #651 [1], where the connection
        would be released if the initial request failed, even if a retry
        succeeded.

        [1] <https://github.com/urllib3/urllib3/issues/651>
        """

        def socket_handler(listener):
            sock = listener.accept()[0]
            consume_socket(sock)
            sock.close()
            rlist, _, _ = select.select([listener], [], [], 5)
            assert rlist
            sock = listener.accept()[0]
            consume_socket(sock)
            sock.send('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nTransfer-Encoding: chunked\r\n\r\n8\r\n12345678\r\n0\r\n\r\n'.encode('utf-8'))
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port, maxsize=1) as pool:
            response = pool.urlopen('GET', '/', retries=1, release_conn=False, preload_content=False, timeout=LONG_TIMEOUT)
            assert pool.num_connections == 2
            assert pool.pool.qsize() == 0
            assert response.connection is not None
            response.read()
            assert pool.pool.qsize() == 1
            assert response.connection is None