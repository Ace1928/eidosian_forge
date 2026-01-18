import ssl
import typing
import trio
from .._exceptions import (
from .base import SOCKET_OPTION, AsyncNetworkBackend, AsyncNetworkStream
def _get_socket_stream(self) -> trio.SocketStream:
    stream = self._stream
    while isinstance(stream, trio.SSLStream):
        stream = stream.transport_stream
    assert isinstance(stream, trio.SocketStream)
    return stream