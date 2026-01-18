from __future__ import absolute_import
import contextlib
import ctypes
import errno
import os.path
import shutil
import socket
import ssl
import struct
import threading
import weakref
import six
from .. import util
from ..util.ssl_ import PROTOCOL_TLS_CLIENT
from ._securetransport.bindings import CoreFoundation, Security, SecurityConst
from ._securetransport.low_level import (
class SecureTransportContext(object):
    """
    I am a wrapper class for the SecureTransport library, to translate the
    interface of the standard library ``SSLContext`` object to calls into
    SecureTransport.
    """

    def __init__(self, protocol):
        self._min_version, self._max_version = _protocol_to_min_max[protocol]
        self._options = 0
        self._verify = False
        self._trust_bundle = None
        self._client_cert = None
        self._client_key = None
        self._client_key_passphrase = None
        self._alpn_protocols = None

    @property
    def check_hostname(self):
        """
        SecureTransport cannot have its hostname checking disabled. For more,
        see the comment on getpeercert() in this file.
        """
        return True

    @check_hostname.setter
    def check_hostname(self, value):
        """
        SecureTransport cannot have its hostname checking disabled. For more,
        see the comment on getpeercert() in this file.
        """
        pass

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, value):
        self._options = value

    @property
    def verify_mode(self):
        return ssl.CERT_REQUIRED if self._verify else ssl.CERT_NONE

    @verify_mode.setter
    def verify_mode(self, value):
        self._verify = True if value == ssl.CERT_REQUIRED else False

    def set_default_verify_paths(self):
        pass

    def load_default_certs(self):
        return self.set_default_verify_paths()

    def set_ciphers(self, ciphers):
        if ciphers != util.ssl_.DEFAULT_CIPHERS:
            raise ValueError("SecureTransport doesn't support custom cipher strings")

    def load_verify_locations(self, cafile=None, capath=None, cadata=None):
        if capath is not None:
            raise ValueError('SecureTransport does not support cert directories')
        if cafile is not None:
            with open(cafile):
                pass
        self._trust_bundle = cafile or cadata

    def load_cert_chain(self, certfile, keyfile=None, password=None):
        self._client_cert = certfile
        self._client_key = keyfile
        self._client_cert_passphrase = password

    def set_alpn_protocols(self, protocols):
        """
        Sets the ALPN protocols that will later be set on the context.

        Raises a NotImplementedError if ALPN is not supported.
        """
        if not hasattr(Security, 'SSLSetALPNProtocols'):
            raise NotImplementedError('SecureTransport supports ALPN only in macOS 10.12+')
        self._alpn_protocols = [six.ensure_binary(p) for p in protocols]

    def wrap_socket(self, sock, server_side=False, do_handshake_on_connect=True, suppress_ragged_eofs=True, server_hostname=None):
        assert not server_side
        assert do_handshake_on_connect
        assert suppress_ragged_eofs
        wrapped_socket = WrappedSocket(sock)
        wrapped_socket.handshake(server_hostname, self._verify, self._trust_bundle, self._min_version, self._max_version, self._client_cert, self._client_key, self._client_key_passphrase, self._alpn_protocols)
        return wrapped_socket