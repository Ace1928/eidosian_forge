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
class LockResult:
    """Result of an operation on a lock; passed to a hook"""

    def __init__(self, lock_url, details=None):
        """Create a lock result for lock with optional details about the lock."""
        self.lock_url = lock_url
        self.details = details

    def __eq__(self, other):
        return self.lock_url == other.lock_url and self.details == other.details

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.lock_url, self.details)