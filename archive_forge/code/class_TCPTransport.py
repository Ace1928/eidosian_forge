import errno
import os
import re
import socket
import ssl
from contextlib import contextmanager
from ssl import SSLError
from struct import pack, unpack
from .exceptions import UnexpectedFrame
from .platform import KNOWN_TCP_OPTS, SOL_TCP
from .utils import set_cloexec
class TCPTransport(_AbstractTransport):
    """Transport that deals directly with TCP socket.

    All parameters are :class:`~amqp.transport._AbstractTransport` class.
    """

    def _setup_transport(self):
        self._write = self.sock.sendall
        self._read_buffer = EMPTY_BUFFER
        self._quick_recv = self.sock.recv

    def _read(self, n, initial=False, _errnos=(errno.EAGAIN, errno.EINTR)):
        """Read exactly n bytes from the socket."""
        recv = self._quick_recv
        rbuf = self._read_buffer
        try:
            while len(rbuf) < n:
                try:
                    s = recv(n - len(rbuf))
                except OSError as exc:
                    if exc.errno in _errnos:
                        if initial and self.raise_on_initial_eintr:
                            raise socket.timeout()
                        continue
                    raise
                if not s:
                    raise OSError('Server unexpectedly closed connection')
                rbuf += s
        except:
            self._read_buffer = rbuf
            raise
        result, self._read_buffer = (rbuf[:n], rbuf[n:])
        return result