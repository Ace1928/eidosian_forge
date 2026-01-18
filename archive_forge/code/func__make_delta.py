from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def _make_delta(self, old):
    """Make an inventory delta from two inventories."""
    if not isinstance(old, CHKInventory):
        return CommonInventory._make_delta(self, old)
    delta = []
    for key, old_value, self_value in self.id_to_entry.iter_changes(old.id_to_entry):
        file_id = key[0]
        if old_value is not None:
            old_path = old.id2path(file_id)
        else:
            old_path = None
        if self_value is not None:
            entry = self._bytes_to_entry(self_value)
            self._fileid_to_entry_cache[file_id] = entry
            new_path = self.id2path(file_id)
        else:
            entry = None
            new_path = None
        delta.append((old_path, new_path, file_id, entry))
    return delta