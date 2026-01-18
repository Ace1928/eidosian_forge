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
class SmartMedium:
    """Base class for smart protocol media, both client- and server-side."""

    def __init__(self):
        self._push_back_buffer = None

    def _push_back(self, data):
        """Return unused bytes to the medium, because they belong to the next
        request(s).

        This sets the _push_back_buffer to the given bytes.
        """
        if not isinstance(data, bytes):
            raise TypeError(data)
        if self._push_back_buffer is not None:
            raise AssertionError('_push_back called when self._push_back_buffer is %r' % (self._push_back_buffer,))
        if data == b'':
            return
        self._push_back_buffer = data

    def _get_push_back_buffer(self):
        if self._push_back_buffer == b'':
            raise AssertionError('%s._push_back_buffer should never be the empty string, which can be confused with EOF' % (self,))
        bytes = self._push_back_buffer
        self._push_back_buffer = None
        return bytes

    def read_bytes(self, desired_count):
        """Read some bytes from this medium.

        :returns: some bytes, possibly more or less than the number requested
            in 'desired_count' depending on the medium.
        """
        if self._push_back_buffer is not None:
            return self._get_push_back_buffer()
        bytes_to_read = min(desired_count, _MAX_READ_SIZE)
        return self._read_bytes(bytes_to_read)

    def _read_bytes(self, count):
        raise NotImplementedError(self._read_bytes)

    def _get_line(self):
        """Read bytes from this request's response until a newline byte.

        This isn't particularly efficient, so should only be used when the
        expected size of the line is quite short.

        :returns: a string of bytes ending in a newline (byte 0x0A).
        """
        line, excess = _get_line(self.read_bytes)
        self._push_back(excess)
        return line

    def _report_activity(self, bytes, direction):
        """Notify that this medium has activity.

        Implementations should call this from all methods that actually do IO.
        Be careful that it's not called twice, if one method is implemented on
        top of another.

        :param bytes: Number of bytes read or written.
        :param direction: 'read' or 'write' or None.
        """
        ui.ui_factory.report_transport_activity(self, bytes, direction)