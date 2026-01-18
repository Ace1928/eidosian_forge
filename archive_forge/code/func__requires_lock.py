import bisect
import codecs
import contextlib
import errno
import operator
import os
import stat
import sys
import time
import zlib
from stat import S_IEXEC
from .. import (cache_utf8, config, debug, errors, lock, osutils, trace,
from . import inventory, static_tuple
from .inventorytree import InventoryTreeChange
def _requires_lock(self):
    """Check that a lock is currently held by someone on the dirstate."""
    if not self._lock_token:
        raise errors.ObjectNotLocked(self)