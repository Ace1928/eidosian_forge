import platform
import select
import socket
import ssl
import sys
import mock
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import SocketDummyServerTestCase, consume_socket
from urllib3.util import ssl_
from urllib3.util.ssltransport import SSLTransport
@pytest.mark.skipif(sys.version_info < (3, 5), reason='requires python3.5 or higher')
class SingleTLSLayerTestCase(SocketDummyServerTestCase):
    """
    Uses the SocketDummyServer to validate a single TLS layer can be
    established through the SSLTransport.
    """

    @classmethod
    def setup_class(cls):
        cls.server_context, cls.client_context = server_client_ssl_contexts()

    def start_dummy_server(self, handler=None):

        def socket_handler(listener):
            sock = listener.accept()[0]
            with self.server_context.wrap_socket(sock, server_side=True) as ssock:
                request = consume_socket(ssock)
                validate_request(request)
                ssock.send(sample_response())
        chosen_handler = handler if handler else socket_handler
        self._start_server(chosen_handler)

    @pytest.mark.timeout(PER_TEST_TIMEOUT)
    def test_start_closed_socket(self):
        """Errors generated from an unconnected socket should bubble up."""
        sock = socket.socket(socket.AF_INET)
        context = ssl.create_default_context()
        sock.close()
        with pytest.raises(OSError):
            SSLTransport(sock, context)

    @pytest.mark.timeout(PER_TEST_TIMEOUT)
    def test_close_after_handshake(self):
        """Socket errors should be bubbled up"""
        self.start_dummy_server()
        sock = socket.create_connection((self.host, self.port))
        with SSLTransport(sock, self.client_context, server_hostname='localhost') as ssock:
            ssock.close()
            with pytest.raises(OSError):
                ssock.send(b'blaaargh')

    @pytest.mark.timeout(PER_TEST_TIMEOUT)
    def test_wrap_existing_socket(self):
        """Validates a single TLS layer can be established."""
        self.start_dummy_server()
        sock = socket.create_connection((self.host, self.port))
        with SSLTransport(sock, self.client_context, server_hostname='localhost') as ssock:
            assert ssock.version() is not None
            ssock.send(sample_request())
            response = consume_socket(ssock)
            validate_response(response)

    @pytest.mark.timeout(PER_TEST_TIMEOUT)
    def test_unbuffered_text_makefile(self):
        self.start_dummy_server()
        sock = socket.create_connection((self.host, self.port))
        with SSLTransport(sock, self.client_context, server_hostname='localhost') as ssock:
            with pytest.raises(ValueError):
                ssock.makefile('r', buffering=0)
            ssock.send(sample_request())
            response = consume_socket(ssock)
            validate_response(response)

    @pytest.mark.timeout(PER_TEST_TIMEOUT)
    def test_unwrap_existing_socket(self):
        """
        Validates we can break up the TLS layer
        A full request/response is sent over TLS, and later over plain text.
        """

        def shutdown_handler(listener):
            sock = listener.accept()[0]
            ssl_sock = self.server_context.wrap_socket(sock, server_side=True)
            request = consume_socket(ssl_sock)
            validate_request(request)
            ssl_sock.sendall(sample_response())
            unwrapped_sock = ssl_sock.unwrap()
            request = consume_socket(unwrapped_sock)
            validate_request(request)
            unwrapped_sock.sendall(sample_response())
        self.start_dummy_server(shutdown_handler)
        sock = socket.create_connection((self.host, self.port))
        ssock = SSLTransport(sock, self.client_context, server_hostname='localhost')
        ssock.sendall(sample_request())
        response = consume_socket(ssock)
        validate_response(response)
        ssock.unwrap()
        sock.sendall(sample_request())
        response = consume_socket(sock)
        validate_response(response)

    @pytest.mark.timeout(PER_TEST_TIMEOUT)
    def test_ssl_object_attributes(self):
        """Ensures common ssl attributes are exposed"""
        self.start_dummy_server()
        sock = socket.create_connection((self.host, self.port))
        with SSLTransport(sock, self.client_context, server_hostname='localhost') as ssock:
            cipher = ssock.cipher()
            assert type(cipher) == tuple
            assert ssock.selected_alpn_protocol() is None
            assert ssock.selected_npn_protocol() is None
            shared_ciphers = ssock.shared_ciphers()
            assert type(shared_ciphers) == list
            assert len(shared_ciphers) > 0
            assert ssock.compression() is None
            validate_peercert(ssock)
            ssock.send(sample_request())
            response = consume_socket(ssock)
            validate_response(response)

    @pytest.mark.timeout(PER_TEST_TIMEOUT)
    def test_socket_object_attributes(self):
        """Ensures common socket attributes are exposed"""
        self.start_dummy_server()
        sock = socket.create_connection((self.host, self.port))
        with SSLTransport(sock, self.client_context, server_hostname='localhost') as ssock:
            assert ssock.fileno() is not None
            test_timeout = 10
            ssock.settimeout(test_timeout)
            assert ssock.gettimeout() == test_timeout
            assert ssock.socket.gettimeout() == test_timeout
            ssock.send(sample_request())
            response = consume_socket(ssock)
            validate_response(response)