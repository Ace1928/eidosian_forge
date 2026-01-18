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
def _find_entry_index(self, key, block):
    """Find the entry index for a key in a block.

        :return: The entry index, True if the entry for the key is present.
        """
    len_block = len(block)
    try:
        if self._last_entry_index is not None:
            entry_index = self._last_entry_index + 1
            if (entry_index > 0 and block[entry_index - 1][0] < key) and key <= block[entry_index][0]:
                self._last_entry_index = entry_index
                present = block[entry_index][0] == key
                return (entry_index, present)
    except IndexError:
        pass
    entry_index = bisect.bisect_left(block, (key, []))
    present = entry_index < len_block and block[entry_index][0] == key
    self._last_entry_index = entry_index
    return (entry_index, present)