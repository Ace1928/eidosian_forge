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
def _iter_child_entries(self, tree_index, path_utf8):
    """Iterate over all the entries that are children of path_utf.

        This only returns entries that are present (not in b'a', b'r') in
        tree_index. tree_index data is not refreshed, so if tree 0 is used,
        results may differ from that obtained if paths were statted to
        determine what ones were directories.

        Asking for the children of a non-directory will return an empty
        iterator.
        """
    pending_dirs = []
    next_pending_dirs = [path_utf8]
    absent = (b'a', b'r')
    while next_pending_dirs:
        pending_dirs = next_pending_dirs
        next_pending_dirs = []
        for path in pending_dirs:
            block_index, present = self._find_block_index_from_key((path, b'', b''))
            if block_index == 0:
                block_index = 1
                if len(self._dirblocks) == 1:
                    return
            if not present:
                continue
            block = self._dirblocks[block_index]
            for entry in block[1]:
                kind = entry[1][tree_index][0]
                if kind not in absent:
                    yield entry
                if kind == b'd':
                    if entry[0][0]:
                        path = entry[0][0] + b'/' + entry[0][1]
                    else:
                        path = entry[0][1]
                    next_pending_dirs.append(path)