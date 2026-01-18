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
def _read_header_if_needed(self):
    """Read the header of the dirstate file if needed."""
    if not self._lock_token:
        raise errors.ObjectNotLocked(self)
    if self._header_state == DirState.NOT_IN_MEMORY:
        self._read_header()