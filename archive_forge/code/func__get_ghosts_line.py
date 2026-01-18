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
def _get_ghosts_line(self, ghost_ids):
    """Create a line for the state file for ghost information."""
    return b'\x00'.join([b'%d' % len(ghost_ids)] + ghost_ids)