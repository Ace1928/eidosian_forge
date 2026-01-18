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
class SmartClientStreamMedium(SmartClientMedium):
    """Stream based medium common class.

    SmartClientStreamMediums operate on a stream. All subclasses use a common
    SmartClientStreamMediumRequest for their requests, and should implement
    _accept_bytes and _read_bytes to allow the request objects to send and
    receive bytes.
    """

    def __init__(self, base):
        SmartClientMedium.__init__(self, base)
        self._current_request = None

    def accept_bytes(self, bytes):
        self._accept_bytes(bytes)

    def __del__(self):
        """The SmartClientStreamMedium knows how to close the stream when it is
        finished with it.
        """
        self.disconnect()

    def _flush(self):
        """Flush the output stream.

        This method is used by the SmartClientStreamMediumRequest to ensure that
        all data for a request is sent, to avoid long timeouts or deadlocks.
        """
        raise NotImplementedError(self._flush)

    def get_request(self):
        """See SmartClientMedium.get_request().

        SmartClientStreamMedium always returns a SmartClientStreamMediumRequest
        for get_request.
        """
        return SmartClientStreamMediumRequest(self)

    def reset(self):
        """We have been disconnected, reset current state.

        This resets things like _current_request and connected state.
        """
        self.disconnect()
        self._current_request = None