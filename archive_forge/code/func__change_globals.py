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
def _change_globals(self):
    from breezy import lockdir, ui
    old_factory = ui.ui_factory
    old_lockdir_timeout = lockdir._DEFAULT_TIMEOUT_SECONDS

    def restore_default_ui_factory_and_lockdir_timeout():
        ui.ui_factory = old_factory
        lockdir._DEFAULT_TIMEOUT_SECONDS = old_lockdir_timeout
    self.cleanups.append(restore_default_ui_factory_and_lockdir_timeout)
    ui.ui_factory = ui.SilentUIFactory()
    lockdir._DEFAULT_TIMEOUT_SECONDS = 0
    orig = signals.install_sighup_handler()

    def restore_signals():
        signals.restore_sighup_handler(orig)
    self.cleanups.append(restore_signals)