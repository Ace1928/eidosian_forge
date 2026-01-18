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
def _get_tcp_socket_defaults(self, sock):
    tcp_opts = {}
    for opt in KNOWN_TCP_OPTS:
        enum = None
        if opt == 'TCP_USER_TIMEOUT':
            try:
                from socket import TCP_USER_TIMEOUT as enum
            except ImportError:
                enum = 18
        elif hasattr(socket, opt):
            enum = getattr(socket, opt)
        if enum:
            if opt in DEFAULT_SOCKET_SETTINGS:
                tcp_opts[enum] = DEFAULT_SOCKET_SETTINGS[opt]
            elif hasattr(socket, opt):
                tcp_opts[enum] = sock.getsockopt(SOL_TCP, getattr(socket, opt))
    return tcp_opts