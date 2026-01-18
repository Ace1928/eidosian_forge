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
def _make_deleted_row(self, fileid_utf8, parents):
    """Return a deleted row for fileid_utf8."""
    return ((b'/', b'RECYCLED.BIN', b'file', fileid_utf8, 0, DirState.NULLSTAT, b''), parents)