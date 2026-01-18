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
def _report_activity(self, bytes, direction):
    """Notify that this medium has activity.

        Implementations should call this from all methods that actually do IO.
        Be careful that it's not called twice, if one method is implemented on
        top of another.

        :param bytes: Number of bytes read or written.
        :param direction: 'read' or 'write' or None.
        """
    ui.ui_factory.report_transport_activity(self, bytes, direction)