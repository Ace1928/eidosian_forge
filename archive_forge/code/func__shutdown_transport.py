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
def _shutdown_transport(self):
    """Unwrap a SSL socket, so we can call shutdown()."""
    if self.sock is not None:
        self.sock = self.sock.unwrap()