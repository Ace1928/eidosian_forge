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
class TestClientCerts(SocketDummyServerTestCase):
    """
    Tests for client certificate support.
    """

    @classmethod
    def setup_class(cls):
        cls.tmpdir = tempfile.mkdtemp()
        ca = trustme.CA()
        cert = ca.issue_cert(u'localhost')
        encrypted_key = encrypt_key_pem(cert.private_key_pem, b'letmein')
        cls.ca_path = os.path.join(cls.tmpdir, 'ca.pem')
        cls.cert_combined_path = os.path.join(cls.tmpdir, 'server.combined.pem')
        cls.cert_path = os.path.join(cls.tmpdir, 'server.pem')
        cls.key_path = os.path.join(cls.tmpdir, 'key.pem')
        cls.password_key_path = os.path.join(cls.tmpdir, 'password_key.pem')
        ca.cert_pem.write_to_path(cls.ca_path)
        cert.private_key_and_cert_chain_pem.write_to_path(cls.cert_combined_path)
        cert.cert_chain_pems[0].write_to_path(cls.cert_path)
        cert.private_key_pem.write_to_path(cls.key_path)
        encrypted_key.write_to_path(cls.password_key_path)

    def teardown_class(cls):
        shutil.rmtree(cls.tmpdir)

    def _wrap_in_ssl(self, sock):
        """
        Given a single socket, wraps it in TLS.
        """
        return ssl.wrap_socket(sock, ssl_version=ssl.PROTOCOL_SSLv23, cert_reqs=ssl.CERT_REQUIRED, ca_certs=self.ca_path, certfile=self.cert_path, keyfile=self.key_path, server_side=True)

    def test_client_certs_two_files(self):
        """
        Having a client cert in a separate file to its associated key works
        properly.
        """
        done_receiving = Event()
        client_certs = []

        def socket_handler(listener):
            sock = listener.accept()[0]
            sock = self._wrap_in_ssl(sock)
            client_certs.append(sock.getpeercert())
            data = b''
            while not data.endswith(b'\r\n\r\n'):
                data += sock.recv(8192)
            sock.sendall(b'HTTP/1.1 200 OK\r\nServer: testsocket\r\nConnection: close\r\nContent-Length: 6\r\n\r\nValid!')
            done_receiving.wait(5)
            sock.close()
        self._start_server(socket_handler)
        with HTTPSConnectionPool(self.host, self.port, cert_file=self.cert_path, key_file=self.key_path, cert_reqs='REQUIRED', ca_certs=self.ca_path) as pool:
            pool.request('GET', '/', retries=0)
            done_receiving.set()
            assert len(client_certs) == 1

    def test_client_certs_one_file(self):
        """
        Having a client cert and its associated private key in just one file
        works properly.
        """
        done_receiving = Event()
        client_certs = []

        def socket_handler(listener):
            sock = listener.accept()[0]
            sock = self._wrap_in_ssl(sock)
            client_certs.append(sock.getpeercert())
            data = b''
            while not data.endswith(b'\r\n\r\n'):
                data += sock.recv(8192)
            sock.sendall(b'HTTP/1.1 200 OK\r\nServer: testsocket\r\nConnection: close\r\nContent-Length: 6\r\n\r\nValid!')
            done_receiving.wait(5)
            sock.close()
        self._start_server(socket_handler)
        with HTTPSConnectionPool(self.host, self.port, cert_file=self.cert_combined_path, cert_reqs='REQUIRED', ca_certs=self.ca_path) as pool:
            pool.request('GET', '/', retries=0)
            done_receiving.set()
            assert len(client_certs) == 1

    def test_missing_client_certs_raises_error(self):
        """
        Having client certs not be present causes an error.
        """
        done_receiving = Event()

        def socket_handler(listener):
            sock = listener.accept()[0]
            try:
                self._wrap_in_ssl(sock)
            except ssl.SSLError:
                pass
            done_receiving.wait(5)
            sock.close()
        self._start_server(socket_handler)
        with HTTPSConnectionPool(self.host, self.port, cert_reqs='REQUIRED', ca_certs=self.ca_path) as pool:
            with pytest.raises(MaxRetryError):
                pool.request('GET', '/', retries=0)
                done_receiving.set()
            done_receiving.set()

    @requires_ssl_context_keyfile_password
    def test_client_cert_with_string_password(self):
        self.run_client_cert_with_password_test(u'letmein')

    @requires_ssl_context_keyfile_password
    def test_client_cert_with_bytes_password(self):
        self.run_client_cert_with_password_test(b'letmein')

    def run_client_cert_with_password_test(self, password):
        """
        Tests client certificate password functionality
        """
        done_receiving = Event()
        client_certs = []

        def socket_handler(listener):
            sock = listener.accept()[0]
            sock = self._wrap_in_ssl(sock)
            client_certs.append(sock.getpeercert())
            data = b''
            while not data.endswith(b'\r\n\r\n'):
                data += sock.recv(8192)
            sock.sendall(b'HTTP/1.1 200 OK\r\nServer: testsocket\r\nConnection: close\r\nContent-Length: 6\r\n\r\nValid!')
            done_receiving.wait(5)
            sock.close()
        self._start_server(socket_handler)
        ssl_context = ssl_.SSLContext(ssl_.PROTOCOL_SSLv23)
        ssl_context.load_cert_chain(certfile=self.cert_path, keyfile=self.password_key_path, password=password)
        with HTTPSConnectionPool(self.host, self.port, ssl_context=ssl_context, cert_reqs='REQUIRED', ca_certs=self.ca_path) as pool:
            pool.request('GET', '/', retries=0)
            done_receiving.set()
            assert len(client_certs) == 1

    @requires_ssl_context_keyfile_password
    def test_load_keyfile_with_invalid_password(self):
        context = ssl_.SSLContext(ssl_.PROTOCOL_SSLv23)
        if ssl_.IS_PYOPENSSL:
            from OpenSSL.SSL import Error
            expected_error = Error
        else:
            expected_error = ssl.SSLError
        with pytest.raises(expected_error):
            context.load_cert_chain(certfile=self.cert_path, keyfile=self.password_key_path, password=b'letmei')