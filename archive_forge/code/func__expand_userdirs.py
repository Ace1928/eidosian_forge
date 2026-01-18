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
def _expand_userdirs(self, path):
    """Translate /~/ or /~user/ to e.g. /home/foo, using
        self.userdir_expander (os.path.expanduser by default).

        If the translated path would fall outside base_path, or the path does
        not start with ~, then no translation is applied.

        If the path is inside, it is adjusted to be relative to the base path.

        e.g. if base_path is /home, and the expanded path is /home/joe, then
        the translated path is joe.
        """
    result = path
    if path.startswith('~'):
        expanded = self.userdir_expander(path)
        if not expanded.endswith('/'):
            expanded += '/'
        if expanded.startswith(self.base_path):
            result = expanded[len(self.base_path):]
    return result