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
def _make_absent(self, current_old):
    """Mark current_old - an entry - as absent for tree 0.

        :return: True if this was the last details entry for the entry key:
            that is, if the underlying block has had the entry removed, thus
            shrinking in length.
        """
    all_remaining_keys = set()
    for details in current_old[1][1:]:
        if details[0] not in (b'a', b'r'):
            all_remaining_keys.add(current_old[0])
        elif details[0] == b'r':
            all_remaining_keys.add(tuple(osutils.split(details[1])) + (current_old[0][2],))
    last_reference = current_old[0] not in all_remaining_keys
    if last_reference:
        block = self._find_block(current_old[0])
        entry_index, present = self._find_entry_index(current_old[0], block[1])
        if not present:
            raise AssertionError('could not find entry for {}'.format(current_old))
        block[1].pop(entry_index)
        if self._id_index is not None:
            self._remove_from_id_index(self._id_index, current_old[0])
    for update_key in all_remaining_keys:
        update_block_index, present = self._find_block_index_from_key(update_key)
        if not present:
            raise AssertionError('could not find block for {}'.format(update_key))
        update_entry_index, present = self._find_entry_index(update_key, self._dirblocks[update_block_index][1])
        if not present:
            raise AssertionError('could not find entry for {}'.format(update_key))
        update_tree_details = self._dirblocks[update_block_index][1][update_entry_index][1]
        if update_tree_details[0][0] == b'a':
            raise AssertionError('bad row {!r}'.format(update_tree_details))
        update_tree_details[0] = DirState.NULL_PARENT_DETAILS
    self._mark_modified()
    return last_reference