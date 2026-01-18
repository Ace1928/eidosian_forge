import sys
from abc import ABC
from asyncio import IncompleteReadError, StreamReader, TimeoutError
from typing import List, Optional, Union
from ..exceptions import (
from ..typing import EncodableT
from .encoders import Encoder
from .socket import SERVER_CLOSED_CONNECTION_ERROR, SocketBuffer
class _RESPBase(BaseParser):
    """Base class for sync-based resp parsing"""

    def __init__(self, socket_read_size):
        self.socket_read_size = socket_read_size
        self.encoder = None
        self._sock = None
        self._buffer = None

    def __del__(self):
        try:
            self.on_disconnect()
        except Exception:
            pass

    def on_connect(self, connection):
        """Called when the socket connects"""
        self._sock = connection._sock
        self._buffer = SocketBuffer(self._sock, self.socket_read_size, connection.socket_timeout)
        self.encoder = connection.encoder

    def on_disconnect(self):
        """Called when the socket disconnects"""
        self._sock = None
        if self._buffer is not None:
            self._buffer.close()
            self._buffer = None
        self.encoder = None

    def can_read(self, timeout):
        return self._buffer and self._buffer.can_read(timeout)