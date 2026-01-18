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
def _maybe_fdatasync(self):
    """Flush to disk if possible and if not configured off."""
    if self._config_stack.get('dirstate.fdatasync'):
        osutils.fdatasync(self._state_file.fileno())