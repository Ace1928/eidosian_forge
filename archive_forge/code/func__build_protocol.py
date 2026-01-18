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
def _build_protocol(self):
    """Identifies the version of the incoming request, and returns an
        a protocol object that can interpret it.

        If more bytes than the version prefix of the request are read, they will
        be fed into the protocol before it is returned.

        :returns: a SmartServerRequestProtocol.
        """
    self._wait_for_bytes_with_timeout(self._client_timeout)
    if self.finished:
        return None
    bytes = self._get_line()
    protocol_factory, unused_bytes = _get_protocol_factory_for_bytes(bytes)
    protocol = protocol_factory(self.backing_transport, self._write_out, self.root_client_path)
    protocol.accept_bytes(unused_bytes)
    return protocol