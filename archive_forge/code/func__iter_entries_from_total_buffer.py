import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _iter_entries_from_total_buffer(self, keys):
    """Iterate over keys when the entire index is parsed."""
    nodes = self._nodes
    keys = [key for key in keys if key in nodes]
    if self.node_ref_lists:
        for key in keys:
            value, node_refs = nodes[key]
            yield (self, key, value, node_refs)
    else:
        for key in keys:
            yield (self, key, nodes[key])