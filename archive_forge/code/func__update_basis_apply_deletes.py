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
def _update_basis_apply_deletes(self, deletes):
    """Apply a sequence of deletes to tree 1 during update_basis_by_delta.

        They may be deletes, or renames that have been split into add/delete
        pairs.

        :param deletes: A sequence of deletes. Each delete is a tuple:
            (old_path_utf8, new_path_utf8, file_id, None, real_delete).
            real_delete is True when the desired outcome is an actual deletion
            rather than the rename handling logic temporarily deleting a path
            during the replacement of a parent.
        """
    null = DirState.NULL_PARENT_DETAILS
    for old_path, new_path, file_id, _, real_delete in deletes:
        if real_delete != (new_path is None):
            self._raise_invalid(old_path, file_id, 'bad delete delta')
        dirname, basename = osutils.split(old_path)
        block_index, entry_index, dir_present, file_present = self._get_block_entry_index(dirname, basename, 1)
        if not file_present:
            self._raise_invalid(old_path, file_id, 'basis tree does not contain removed entry')
        entry = self._dirblocks[block_index][1][entry_index]
        active_kind = entry[1][0][0]
        if entry[0][2] != file_id:
            self._raise_invalid(old_path, file_id, 'mismatched file_id in tree 1')
        dir_block = ()
        old_kind = entry[1][1][0]
        if active_kind in b'ar':
            if active_kind == b'r':
                active_path = entry[1][0][1]
                active_entry = self._get_entry(0, file_id, active_path)
                if active_entry[1][1][0] != b'r':
                    self._raise_invalid(old_path, file_id, 'Dirstate did not have matching rename entries')
                elif active_entry[1][0][0] in b'ar':
                    self._raise_invalid(old_path, file_id, 'Dirstate had a rename pointing at an inactive tree0')
                active_entry[1][1] = null
            del self._dirblocks[block_index][1][entry_index]
            if old_kind == b'd':
                dir_block_index, present = self._find_block_index_from_key((old_path, b'', b''))
                if present:
                    dir_block = self._dirblocks[dir_block_index][1]
                    if not dir_block:
                        del self._dirblocks[dir_block_index]
        else:
            entry[1][1] = null
            block_i, entry_i, d_present, f_present = self._get_block_entry_index(old_path, b'', 1)
            if d_present:
                dir_block = self._dirblocks[block_i][1]
        for child_entry in dir_block:
            child_basis_kind = child_entry[1][1][0]
            if child_basis_kind not in b'ar':
                self._raise_invalid(old_path, file_id, 'The file id was deleted but its children were not deleted.')