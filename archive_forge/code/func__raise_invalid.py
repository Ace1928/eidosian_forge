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
def _raise_invalid(self, path, file_id, reason):
    self._changes_aborted = True
    raise errors.InconsistentDelta(path, file_id, reason)