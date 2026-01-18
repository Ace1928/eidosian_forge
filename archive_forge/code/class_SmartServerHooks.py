import errno
import os.path
import socket
import sys
import threading
import time
from ... import errors, trace
from ... import transport as _mod_transport
from ...hooks import Hooks
from ...i18n import gettext
from ...lazy_import import lazy_import
from breezy.bzr.smart import (
from breezy.transport import (
from breezy import (
class SmartServerHooks(Hooks):
    """Hooks for the smart server."""

    def __init__(self):
        """Create the default hooks.

        These are all empty initially, because by default nothing should get
        notified.
        """
        Hooks.__init__(self, 'breezy.bzr.smart.server', 'SmartTCPServer.hooks')
        self.add_hook('server_started', 'Called by the bzr server when it starts serving a directory. server_started is called with (backing urls, public url), where backing_url is a list of URLs giving the server-specific directory locations, and public_url is the public URL for the directory being served.', (0, 16))
        self.add_hook('server_started_ex', 'Called by the bzr server when it starts serving a directory. server_started is called with (backing_urls, server_obj).', (1, 17))
        self.add_hook('server_stopped', 'Called by the bzr server when it stops serving a directory. server_stopped is called with the same parameters as the server_started hook: (backing_urls, public_url).', (0, 16))
        self.add_hook('server_exception', 'Called by the bzr server when an exception occurs. server_exception is called with the sys.exc_info() tuple return true for the hook if the exception has been handled, in which case the server will exit normally.', (2, 4))