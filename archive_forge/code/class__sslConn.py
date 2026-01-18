from __future__ import annotations
import socket as _socket
import ssl as _stdlibssl
import sys as _sys
import time as _time
from errno import EINTR as _EINTR
from ipaddress import ip_address as _ip_address
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union
from cryptography.x509 import load_der_x509_certificate as _load_der_x509_certificate
from OpenSSL import SSL as _SSL
from OpenSSL import crypto as _crypto
from service_identity import CertificateError as _SICertificateError
from service_identity import VerificationError as _SIVerificationError
from service_identity.pyopenssl import verify_hostname as _verify_hostname
from service_identity.pyopenssl import verify_ip_address as _verify_ip_address
from pymongo.errors import ConfigurationError as _ConfigurationError
from pymongo.errors import _CertificateError
from pymongo.ocsp_cache import _OCSPCache
from pymongo.ocsp_support import _load_trusted_ca_certs, _ocsp_callback
from pymongo.socket_checker import SocketChecker as _SocketChecker
from pymongo.socket_checker import _errno_from_exception
from pymongo.write_concern import validate_boolean
class _sslConn(_SSL.Connection):

    def __init__(self, ctx: _SSL.Context, sock: Optional[_socket.socket], suppress_ragged_eofs: bool):
        self.socket_checker = _SocketChecker()
        self.suppress_ragged_eofs = suppress_ragged_eofs
        super().__init__(ctx, sock)

    def _call(self, call: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
        timeout = self.gettimeout()
        if timeout:
            start = _time.monotonic()
        while True:
            try:
                return call(*args, **kwargs)
            except BLOCKING_IO_ERRORS as exc:
                if self.fileno() == -1:
                    if timeout and _time.monotonic() - start > timeout:
                        raise _socket.timeout('timed out') from None
                    raise SSLError('Underlying socket has been closed') from None
                if isinstance(exc, _SSL.WantReadError):
                    want_read = True
                    want_write = False
                elif isinstance(exc, _SSL.WantWriteError):
                    want_read = False
                    want_write = True
                else:
                    want_read = True
                    want_write = True
                self.socket_checker.select(self, want_read, want_write, timeout)
                if timeout and _time.monotonic() - start > timeout:
                    raise _socket.timeout('timed out') from None
                continue

    def do_handshake(self, *args: Any, **kwargs: Any) -> None:
        return self._call(super().do_handshake, *args, **kwargs)

    def recv(self, *args: Any, **kwargs: Any) -> bytes:
        try:
            return self._call(super().recv, *args, **kwargs)
        except _SSL.SysCallError as exc:
            if self.suppress_ragged_eofs and _ragged_eof(exc):
                return b''
            raise

    def recv_into(self, *args: Any, **kwargs: Any) -> int:
        try:
            return self._call(super().recv_into, *args, **kwargs)
        except _SSL.SysCallError as exc:
            if self.suppress_ragged_eofs and _ragged_eof(exc):
                return 0
            raise

    def sendall(self, buf: bytes, flags: int=0) -> None:
        view = memoryview(buf)
        total_length = len(buf)
        total_sent = 0
        while total_sent < total_length:
            try:
                sent = self._call(super().send, view[total_sent:], flags)
            except OSError as exc:
                if _errno_from_exception(exc) == _EINTR:
                    continue
                raise
            if sent <= 0:
                raise OSError('connection closed')
            total_sent += sent