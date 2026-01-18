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
def _iter_specific_file_parents(self):
    """Iter over the specific file parents."""
    while self.search_specific_file_parents:
        path_utf8 = self.search_specific_file_parents.pop()
        if osutils.is_inside_any(self.searched_specific_files, path_utf8):
            continue
        if path_utf8 in self.searched_exact_paths:
            continue
        path_entries = self.state._entries_for_path(path_utf8)
        selected_entries = []
        found_item = False
        for candidate_entry in path_entries:
            if candidate_entry[1][self.target_index][0] not in (b'a', b'r'):
                found_item = True
                selected_entries.append(candidate_entry)
            elif self.source_index is not None and candidate_entry[1][self.source_index][0] not in (b'a', b'r'):
                found_item = True
                if candidate_entry[1][self.target_index][0] == b'a':
                    selected_entries.append(candidate_entry)
                else:
                    self.search_specific_file_parents.add(candidate_entry[1][self.target_index][1])
        if not found_item:
            raise AssertionError('Missing entry for specific path parent {!r}, {!r}'.format(path_utf8, path_entries))
        path_info = self._path_info(path_utf8, path_utf8.decode('utf8'))
        for entry in selected_entries:
            if entry[0][2] in self.seen_ids:
                continue
            result, changed = self._process_entry(entry, path_info)
            if changed is None:
                raise AssertionError('Got entry<->path mismatch for specific path %r entry %r path_info %r ' % (path_utf8, entry, path_info))
            if changed:
                self._gather_result_for_consistency(result)
                if result.kind[0] == 'directory' and result.kind[1] != 'directory':
                    if entry[1][self.source_index][0] == b'r':
                        entry_path_utf8 = entry[1][self.source_index][1]
                    else:
                        entry_path_utf8 = path_utf8
                    initial_key = (entry_path_utf8, b'', b'')
                    block_index, _ = self.state._find_block_index_from_key(initial_key)
                    if block_index == 0:
                        block_index += 1
                    current_block = None
                    if block_index < len(self.state._dirblocks):
                        current_block = self.state._dirblocks[block_index]
                        if not osutils.is_inside(entry_path_utf8, current_block[0]):
                            current_block = None
                    if current_block is not None:
                        for entry in current_block[1]:
                            if entry[1][self.source_index][0] in (b'a', b'r'):
                                continue
                            self.search_specific_file_parents.add(osutils.pathjoin(*entry[0][:2]))
            if changed or self.include_unchanged:
                yield result
        self.searched_exact_paths.add(path_utf8)