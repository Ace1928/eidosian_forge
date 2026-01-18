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
def _init_socket(self, socket_settings, read_timeout, write_timeout):
    self.sock.settimeout(None)
    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    self._set_socket_options(socket_settings)
    for timeout, interval in ((socket.SO_SNDTIMEO, write_timeout), (socket.SO_RCVTIMEO, read_timeout)):
        if interval is not None:
            sec = int(interval)
            usec = int((interval - sec) * 1000000)
            self.sock.setsockopt(socket.SOL_SOCKET, timeout, pack('ll', sec, usec))
    self._setup_transport()
    self._write(AMQP_PROTOCOL_HEADER)