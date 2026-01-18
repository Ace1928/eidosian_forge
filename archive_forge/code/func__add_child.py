from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def _add_child(self, entry):
    """Add an entry to the inventory, without adding it to its parent"""
    if entry.file_id in self._byid:
        raise errors.BzrError('inventory already contains entry with id {%s}' % entry.file_id)
    self._byid[entry.file_id] = entry
    children = getattr(entry, 'children', {})
    if children is not None:
        for child in children.values():
            self._add_child(child)
    return entry