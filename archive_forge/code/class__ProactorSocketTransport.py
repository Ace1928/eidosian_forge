import io
import os
import socket
import warnings
import signal
import threading
import collections
from . import base_events
from . import constants
from . import futures
from . import exceptions
from . import protocols
from . import sslproto
from . import transports
from . import trsock
from .log import logger
class _ProactorSocketTransport(_ProactorReadPipeTransport, _ProactorBaseWritePipeTransport, transports.Transport):
    """Transport for connected sockets."""
    _sendfile_compatible = constants._SendfileMode.TRY_NATIVE

    def __init__(self, loop, sock, protocol, waiter=None, extra=None, server=None):
        super().__init__(loop, sock, protocol, waiter, extra, server)
        base_events._set_nodelay(sock)

    def _set_extra(self, sock):
        _set_socket_extra(self, sock)

    def can_write_eof(self):
        return True

    def write_eof(self):
        if self._closing or self._eof_written:
            return
        self._eof_written = True
        if self._write_fut is None:
            self._sock.shutdown(socket.SHUT_WR)