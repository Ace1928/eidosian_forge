import socket
import ssl
import sys
import typing
from functools import partial
from .._exceptions import (
from .._utils import is_socket_readable
from .base import SOCKET_OPTION, NetworkBackend, NetworkStream
class SyncStream(NetworkStream):

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock

    def read(self, max_bytes: int, timeout: typing.Optional[float]=None) -> bytes:
        exc_map: ExceptionMapping = {socket.timeout: ReadTimeout, OSError: ReadError}
        with map_exceptions(exc_map):
            self._sock.settimeout(timeout)
            return self._sock.recv(max_bytes)

    def write(self, buffer: bytes, timeout: typing.Optional[float]=None) -> None:
        if not buffer:
            return
        exc_map: ExceptionMapping = {socket.timeout: WriteTimeout, OSError: WriteError}
        with map_exceptions(exc_map):
            while buffer:
                self._sock.settimeout(timeout)
                n = self._sock.send(buffer)
                buffer = buffer[n:]

    def close(self) -> None:
        self._sock.close()

    def start_tls(self, ssl_context: ssl.SSLContext, server_hostname: typing.Optional[str]=None, timeout: typing.Optional[float]=None) -> NetworkStream:
        exc_map: ExceptionMapping = {socket.timeout: ConnectTimeout, OSError: ConnectError}
        with map_exceptions(exc_map):
            try:
                if isinstance(self._sock, ssl.SSLSocket):
                    return TLSinTLSStream(self._sock, ssl_context, server_hostname, timeout)
                else:
                    self._sock.settimeout(timeout)
                    sock = ssl_context.wrap_socket(self._sock, server_hostname=server_hostname)
            except Exception as exc:
                self.close()
                raise exc
        return SyncStream(sock)

    def get_extra_info(self, info: str) -> typing.Any:
        if info == 'ssl_object' and isinstance(self._sock, ssl.SSLSocket):
            return self._sock._sslobj
        if info == 'client_addr':
            return self._sock.getsockname()
        if info == 'server_addr':
            return self._sock.getpeername()
        if info == 'socket':
            return self._sock
        if info == 'is_readable':
            return is_socket_readable(self._sock)
        return None