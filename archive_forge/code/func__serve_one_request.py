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
def _serve_one_request(self, protocol):
    """Read one request from input, process, send back a response.

        :param protocol: a SmartServerRequestProtocol.
        """
    if protocol is None:
        return
    try:
        self._serve_one_request_unguarded(protocol)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        self.terminate_due_to_error()