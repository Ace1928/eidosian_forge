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
def _read_dirblocks_if_needed(self):
    """Read in all the dirblocks from the file if they are not in memory.

        This populates self._dirblocks, and sets self._dirblock_state to
        IN_MEMORY_UNMODIFIED. It is not currently ready for incremental block
        loading.
        """
    self._read_header_if_needed()
    if self._dirblock_state == DirState.NOT_IN_MEMORY:
        _read_dirblocks(self)