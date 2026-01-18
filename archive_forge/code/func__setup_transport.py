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
def _setup_transport(self):
    self._write = self.sock.sendall
    self._read_buffer = EMPTY_BUFFER
    self._quick_recv = self.sock.recv