from __future__ import absolute_import
import OpenSSL.SSL
from cryptography import x509
from cryptography.hazmat.backends.openssl import backend as openssl_backend
from cryptography.hazmat.backends.openssl.x509 import _Certificate
from io import BytesIO
from socket import error as SocketError
from socket import timeout
import logging
import ssl
import sys
from .. import util
from ..packages import six
from ..util.ssl_ import PROTOCOL_TLS_CLIENT
class PyOpenSSLContext(object):
    """
    I am a wrapper class for the PyOpenSSL ``Context`` object. I am responsible
    for translating the interface of the standard library ``SSLContext`` object
    to calls into PyOpenSSL.
    """

    def __init__(self, protocol):
        self.protocol = _openssl_versions[protocol]
        self._ctx = OpenSSL.SSL.Context(self.protocol)
        self._options = 0
        self.check_hostname = False

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, value):
        self._options = value
        self._ctx.set_options(value)

    @property
    def verify_mode(self):
        return _openssl_to_stdlib_verify[self._ctx.get_verify_mode()]

    @verify_mode.setter
    def verify_mode(self, value):
        self._ctx.set_verify(_stdlib_to_openssl_verify[value], _verify_callback)

    def set_default_verify_paths(self):
        self._ctx.set_default_verify_paths()

    def set_ciphers(self, ciphers):
        if isinstance(ciphers, six.text_type):
            ciphers = ciphers.encode('utf-8')
        self._ctx.set_cipher_list(ciphers)

    def load_verify_locations(self, cafile=None, capath=None, cadata=None):
        if cafile is not None:
            cafile = cafile.encode('utf-8')
        if capath is not None:
            capath = capath.encode('utf-8')
        try:
            self._ctx.load_verify_locations(cafile, capath)
            if cadata is not None:
                self._ctx.load_verify_locations(BytesIO(cadata))
        except OpenSSL.SSL.Error as e:
            raise ssl.SSLError('unable to load trusted certificates: %r' % e)

    def load_cert_chain(self, certfile, keyfile=None, password=None):
        self._ctx.use_certificate_chain_file(certfile)
        if password is not None:
            if not isinstance(password, six.binary_type):
                password = password.encode('utf-8')
            self._ctx.set_passwd_cb(lambda *_: password)
        self._ctx.use_privatekey_file(keyfile or certfile)

    def set_alpn_protocols(self, protocols):
        protocols = [six.ensure_binary(p) for p in protocols]
        return self._ctx.set_alpn_protos(protocols)

    def wrap_socket(self, sock, server_side=False, do_handshake_on_connect=True, suppress_ragged_eofs=True, server_hostname=None):
        cnx = OpenSSL.SSL.Connection(self._ctx, sock)
        if isinstance(server_hostname, six.text_type):
            server_hostname = server_hostname.encode('utf-8')
        if server_hostname is not None:
            cnx.set_tlsext_host_name(server_hostname)
        cnx.set_connect_state()
        while True:
            try:
                cnx.do_handshake()
            except OpenSSL.SSL.WantReadError:
                if not util.wait_for_read(sock, sock.gettimeout()):
                    raise timeout('select timed out')
                continue
            except OpenSSL.SSL.Error as e:
                raise ssl.SSLError('bad handshake: %r' % e)
            break
        return WrappedSocket(cnx, sock)