import socket
import selectors
import os
import sys
import threading
from io import BufferedIOBase
from time import monotonic as time
class _SocketWriter(BufferedIOBase):
    """Simple writable BufferedIOBase implementation for a socket

    Does not hold data in a buffer, avoiding any need to call flush()."""

    def __init__(self, sock):
        self._sock = sock

    def writable(self):
        return True

    def write(self, b):
        self._sock.sendall(b)
        with memoryview(b) as view:
            return view.nbytes

    def fileno(self):
        return self._sock.fileno()