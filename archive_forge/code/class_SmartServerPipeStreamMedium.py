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
class SmartServerPipeStreamMedium(SmartServerStreamMedium):

    def __init__(self, in_file, out_file, backing_transport, timeout=None):
        """Construct new server.

        :param in_file: Python file from which requests can be read.
        :param out_file: Python file to write responses.
        :param backing_transport: Transport for the directory served.
        """
        SmartServerStreamMedium.__init__(self, backing_transport, timeout=timeout)
        if sys.platform == 'win32':
            import msvcrt
            for f in (in_file, out_file):
                fileno = getattr(f, 'fileno', None)
                if fileno:
                    msvcrt.setmode(fileno(), os.O_BINARY)
        self._in = in_file
        self._out = out_file

    def serve(self):
        """See SmartServerStreamMedium.serve"""
        stop_gracefully = self._stop_gracefully
        signals.register_on_hangup(id(self), stop_gracefully)
        try:
            return super().serve()
        finally:
            signals.unregister_on_hangup(id(self))

    def _serve_one_request_unguarded(self, protocol):
        while True:
            bytes_to_read = protocol.next_read_size()
            if bytes_to_read == 0:
                self._out.flush()
                return
            bytes = self.read_bytes(bytes_to_read)
            if bytes == b'':
                self.finished = True
                self._out.flush()
                return
            protocol.accept_bytes(bytes)

    def _disconnect_client(self):
        self._in.close()
        self._out.flush()
        self._out.close()

    def _wait_for_bytes_with_timeout(self, timeout_seconds):
        """Wait for more bytes to be read, but timeout if none available.

        This allows us to detect idle connections, and stop trying to read from
        them, without setting the socket itself to non-blocking. This also
        allows us to specify when we watch for idle timeouts.

        :return: None, this will raise ConnectionTimeout if we time out before
            data is available.
        """
        if getattr(self._in, 'fileno', None) is None or sys.platform == 'win32':
            return
        try:
            return self._wait_on_descriptor(self._in, timeout_seconds)
        except io.UnsupportedOperation:
            return

    def _read_bytes(self, desired_count):
        return self._in.read(desired_count)

    def terminate_due_to_error(self):
        self._out.close()
        self.finished = True

    def _write_out(self, bytes):
        self._out.write(bytes)