import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _move_to_front_by_index(self, hit_indices):
    """Core logic for _move_to_front.

        Returns a list of names corresponding to the hit_indices param.
        """
    indices_info = zip(self._index_names, self._indices)
    if 'index' in debug.debug_flags:
        indices_info = list(indices_info)
        trace.mutter('CombinedGraphIndex reordering: currently %r, promoting %r', indices_info, hit_indices)
    hit_names = []
    unhit_names = []
    new_hit_indices = []
    unhit_indices = []
    for offset, (name, idx) in enumerate(indices_info):
        if idx in hit_indices:
            hit_names.append(name)
            new_hit_indices.append(idx)
            if len(new_hit_indices) == len(hit_indices):
                unhit_names.extend(self._index_names[offset + 1:])
                unhit_indices.extend(self._indices[offset + 1:])
                break
        else:
            unhit_names.append(name)
            unhit_indices.append(idx)
    self._indices = new_hit_indices + unhit_indices
    self._index_names = hit_names + unhit_names
    if 'index' in debug.debug_flags:
        trace.mutter('CombinedGraphIndex reordered: %r', self._indices)
    return hit_names