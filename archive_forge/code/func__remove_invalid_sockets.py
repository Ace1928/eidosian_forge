import io
import os
import socket
import threading
import time
import selectors
from contextlib import suppress
from . import errors
from ._compat import IS_WINDOWS
from .makefile import MakeFile
def _remove_invalid_sockets(self):
    """Clean up the resources of any broken connections.

        This method attempts to detect any connections in an invalid state,
        unregisters them from the selector and closes the file descriptors of
        the corresponding network sockets where possible.
        """
    invalid_conns = []
    for sock_fd, conn in self._selector.connections:
        if conn is self.server:
            continue
        try:
            os.fstat(sock_fd)
        except OSError:
            invalid_conns.append((sock_fd, conn))
    for sock_fd, conn in invalid_conns:
        self._selector.unregister(sock_fd)
        with suppress(OSError):
            conn.close()