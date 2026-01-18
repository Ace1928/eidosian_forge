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
def _entries_for_path(self, path):
    """Return a list with all the entries that match path for all ids."""
    dirname, basename = os.path.split(path)
    key = (dirname, basename, b'')
    block_index, present = self._find_block_index_from_key(key)
    if not present:
        return []
    result = []
    block = self._dirblocks[block_index][1]
    entry_index, _ = self._find_entry_index(key, block)
    while entry_index < len(block) and block[entry_index][0][0:2] == key[0:2]:
        result.append(block[entry_index])
        entry_index += 1
    return result