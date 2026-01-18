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
class SmartServerSocketStreamMedium(SmartServerStreamMedium):

    def __init__(self, sock, backing_transport, root_client_path='/', timeout=None):
        """Constructor.

        :param sock: the socket the server will read from.  It will be put
            into blocking mode.
        """
        SmartServerStreamMedium.__init__(self, backing_transport, root_client_path=root_client_path, timeout=timeout)
        sock.setblocking(True)
        self.socket = sock
        try:
            self._client_info = sock.getpeername()
        except OSError:
            self._client_info = '<unknown>'

    def __str__(self):
        return '{}(client={})'.format(self.__class__.__name__, self._client_info)

    def __repr__(self):
        return '{}.{}(client={})'.format(self.__module__, self.__class__.__name__, self._client_info)

    def _serve_one_request_unguarded(self, protocol):
        while protocol.next_read_size():
            bytes = self.read_bytes(osutils.MAX_SOCKET_CHUNK)
            if bytes == b'':
                self.finished = True
                return
            protocol.accept_bytes(bytes)
        self._push_back(protocol.unused_data)

    def _disconnect_client(self):
        """Close the current connection. We stopped due to a timeout/etc."""
        self.socket.close()

    def _wait_for_bytes_with_timeout(self, timeout_seconds):
        """Wait for more bytes to be read, but timeout if none available.

        This allows us to detect idle connections, and stop trying to read from
        them, without setting the socket itself to non-blocking. This also
        allows us to specify when we watch for idle timeouts.

        :return: None, this will raise ConnectionTimeout if we time out before
            data is available.
        """
        return self._wait_on_descriptor(self.socket, timeout_seconds)

    def _read_bytes(self, desired_count):
        return osutils.read_bytes_from_socket(self.socket, self._report_activity)

    def terminate_due_to_error(self):
        self.socket.close()
        self.finished = True

    def _write_out(self, bytes):
        tstart = osutils.perf_counter()
        osutils.send_all(self.socket, bytes, self._report_activity)
        if 'hpss' in debug.debug_flags:
            thread_id = _thread.get_ident()
            trace.mutter('%12s: [%s] %d bytes to the socket in %.3fs' % ('wrote', thread_id, len(bytes), osutils.perf_counter() - tstart))