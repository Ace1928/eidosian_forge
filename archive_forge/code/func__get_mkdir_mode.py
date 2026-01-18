import contextlib
import sys
from typing import TYPE_CHECKING, Set, cast
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import config, controldir, errors, lockdir
from .. import transport as _mod_transport
from ..trace import mutter, note, warning
from ..transport import do_catching_redirections, local
def _get_mkdir_mode(self):
    """Figure out the mode to use when creating a bzrdir subdir."""
    temp_control = lockable_files.LockableFiles(self.transport, '', lockable_files.TransportLock)
    return temp_control._dir_mode