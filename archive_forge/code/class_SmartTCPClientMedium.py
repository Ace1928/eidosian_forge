import _thread
import errno
import io
import os
import sys
import time
import breezy
from ...lazy_import import lazy_import
import select
import socket
import weakref
from breezy import (
from breezy.i18n import gettext
from breezy.bzr.smart import client, protocol, request, signals, vfs
from breezy.transport import ssh
from ... import errors, osutils
class SmartTCPClientMedium(SmartClientSocketMedium):
    """A client medium that creates a TCP connection."""

    def __init__(self, host, port, base):
        """Creates a client that will connect on the first use."""
        SmartClientSocketMedium.__init__(self, base)
        self._host = host
        self._port = port

    def _ensure_connection(self):
        """Connect this medium if not already connected."""
        if self._connected:
            return
        if self._port is None:
            port = BZR_DEFAULT_PORT
        else:
            port = int(self._port)
        try:
            sockaddrs = socket.getaddrinfo(self._host, port, socket.AF_UNSPEC, socket.SOCK_STREAM, 0, 0)
        except socket.gaierror as xxx_todo_changeme:
            err_num, err_msg = xxx_todo_changeme.args
            raise errors.ConnectionError('failed to lookup %s:%d: %s' % (self._host, port, err_msg))
        last_err = socket.error('no address found for %s' % self._host)
        for family, socktype, proto, canonname, sockaddr in sockaddrs:
            try:
                self._socket = socket.socket(family, socktype, proto)
                self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._socket.connect(sockaddr)
            except OSError as err:
                if self._socket is not None:
                    self._socket.close()
                self._socket = None
                last_err = err
                continue
            break
        if self._socket is None:
            if isinstance(last_err.args, str):
                err_msg = last_err.args
            else:
                err_msg = last_err.args[1]
            raise errors.ConnectionError('failed to connect to %s:%d: %s' % (self._host, port, err_msg))
        self._connected = True
        for hook in transport.Transport.hooks['post_connect']:
            hook(self)