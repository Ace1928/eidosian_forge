import os
import time
import yaml
from . import config, debug, errors, lock, osutils, ui, urlutils
from .decorators import only_raises
from .errors import (DirectoryNotEmpty, LockBreakMismatch, LockBroken,
from .i18n import gettext
from .osutils import format_delta, get_host_name, rand_chars
from .trace import mutter, note
from .transport import FileExists, NoSuchFile
def _check_not_locked(self):
    """If the lock is held by this instance, raise an error."""
    if self._lock_held:
        raise AssertionError("can't break own lock: %r" % self)