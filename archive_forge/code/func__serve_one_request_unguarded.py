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