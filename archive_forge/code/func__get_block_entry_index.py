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
def _get_block_entry_index(self, dirname, basename, tree_index):
    """Get the coordinates for a path in the state structure.

        :param dirname: The utf8 dirname to lookup.
        :param basename: The utf8 basename to lookup.
        :param tree_index: The index of the tree for which this lookup should
            be attempted.
        :return: A tuple describing where the path is located, or should be
            inserted. The tuple contains four fields: the block index, the row
            index, the directory is present (boolean), the entire path is
            present (boolean).  There is no guarantee that either
            coordinate is currently reachable unless the found field for it is
            True. For instance, a directory not present in the searched tree
            may be returned with a value one greater than the current highest
            block offset. The directory present field will always be True when
            the path present field is True. The directory present field does
            NOT indicate that the directory is present in the searched tree,
            rather it indicates that there are at least some files in some
            tree present there.
        """
    self._read_dirblocks_if_needed()
    key = (dirname, basename, b'')
    block_index, present = self._find_block_index_from_key(key)
    if not present:
        return (block_index, 0, False, False)
    block = self._dirblocks[block_index][1]
    entry_index, present = self._find_entry_index(key, block)
    while entry_index < len(block) and block[entry_index][0][1] == basename:
        if block[entry_index][1][tree_index][0] not in (b'a', b'r'):
            return (block_index, entry_index, True, True)
        entry_index += 1
    return (block_index, entry_index, True, False)