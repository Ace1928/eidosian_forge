import io
import os
import sys
import socket
import struct
import time
import tempfile
import itertools
from . import util
from . import AuthenticationError, BufferTooShort
from .context import reduction
class SocketListener(object):
    """
    Representation of a socket which is bound to an address and listening
    """

    def __init__(self, address, family, backlog=1):
        self._socket = socket.socket(getattr(socket, family))
        try:
            if os.name == 'posix':
                self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.setblocking(True)
            self._socket.bind(address)
            self._socket.listen(backlog)
            self._address = self._socket.getsockname()
        except OSError:
            self._socket.close()
            raise
        self._family = family
        self._last_accepted = None
        if family == 'AF_UNIX' and (not util.is_abstract_socket_namespace(address)):
            self._unlink = util.Finalize(self, os.unlink, args=(address,), exitpriority=0)
        else:
            self._unlink = None

    def accept(self):
        s, self._last_accepted = self._socket.accept()
        s.setblocking(True)
        return Connection(s.detach())

    def close(self):
        try:
            self._socket.close()
        finally:
            unlink = self._unlink
            if unlink is not None:
                self._unlink = None
                unlink()