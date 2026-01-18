from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def _parent_id_basename_key(self, entry):
    """Create a key for a entry in a parent_id_basename_to_file_id index."""
    if entry.parent_id is not None:
        parent_id = entry.parent_id
    else:
        parent_id = b''
    return StaticTuple(parent_id, entry.name.encode('utf8')).intern()