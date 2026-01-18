import enum
import logging
import ssl
import time
from types import TracebackType
from typing import (
import h11
from .._backends.base import NetworkStream
from .._exceptions import (
from .._models import Origin, Request, Response
from .._synchronization import Lock, ShieldCancellation
from .._trace import Trace
from .interfaces import ConnectionInterface
class HTTP11UpgradeStream(NetworkStream):

    def __init__(self, stream: NetworkStream, leading_data: bytes) -> None:
        self._stream = stream
        self._leading_data = leading_data

    def read(self, max_bytes: int, timeout: Optional[float]=None) -> bytes:
        if self._leading_data:
            buffer = self._leading_data[:max_bytes]
            self._leading_data = self._leading_data[max_bytes:]
            return buffer
        else:
            return self._stream.read(max_bytes, timeout)

    def write(self, buffer: bytes, timeout: Optional[float]=None) -> None:
        self._stream.write(buffer, timeout)

    def close(self) -> None:
        self._stream.close()

    def start_tls(self, ssl_context: ssl.SSLContext, server_hostname: Optional[str]=None, timeout: Optional[float]=None) -> NetworkStream:
        return self._stream.start_tls(ssl_context, server_hostname, timeout)

    def get_extra_info(self, info: str) -> Any:
        return self._stream.get_extra_info(info)