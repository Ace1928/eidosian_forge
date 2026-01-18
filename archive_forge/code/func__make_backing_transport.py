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
def _make_backing_transport(self, transport):
    """Chroot transport, and decorate with userdir expander."""
    self.base_path = self.get_base_path(transport)
    chroot_server = chroot.ChrootServer(transport)
    chroot_server.start_server()
    self.cleanups.append(chroot_server.stop_server)
    transport = _mod_transport.get_transport_from_url(chroot_server.get_url())
    if self.base_path is not None:
        expand_userdirs = self._make_expand_userdirs_filter(transport)
        expand_userdirs.start_server()
        self.cleanups.append(expand_userdirs.stop_server)
        transport = _mod_transport.get_transport_from_url(expand_userdirs.get_url())
    self.transport = transport