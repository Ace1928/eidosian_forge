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
def _note_lock(self, lock_type):
    if 'relock' in debug.debug_flags and self._prev_lock == lock_type:
        if lock_type == 'r':
            type_name = 'read'
        else:
            type_name = 'write'
        trace.note(gettext('{0!r} was {1} locked again'), self, type_name)
    self._prev_lock = lock_type