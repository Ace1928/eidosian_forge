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
def _check_delta_ids_absent(self, new_ids, delta, tree_index):
    """Check that none of the file_ids in new_ids are present in a tree."""
    if not new_ids:
        return
    id_index = self._get_id_index()
    for file_id in new_ids:
        for key in id_index.get(file_id, ()):
            block_i, entry_i, d_present, f_present = self._get_block_entry_index(key[0], key[1], tree_index)
            if not f_present:
                continue
            entry = self._dirblocks[block_i][1][entry_i]
            if entry[0][2] != file_id:
                continue
            self._raise_invalid((b'%s/%s' % key[0:2]).decode('utf8'), file_id, 'This file_id is new in the delta but already present in the target')