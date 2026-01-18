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
def _finished_reading(self):
    """See SmartClientMediumRequest._finished_reading.

        This clears the _current_request on self._medium to allow a new
        request to be created.
        """
    if self._medium._current_request is not self:
        raise AssertionError()
    self._medium._current_request = None