import contextlib
import errno
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple
from . import debug, errors, osutils, trace
from .hooks import Hooks
from .i18n import gettext
from .transport import Transport
class _fcntl_FileLock(_OSLock):

    def _unlock(self):
        fcntl.lockf(self.f, fcntl.LOCK_UN)
        self._clear_f()