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