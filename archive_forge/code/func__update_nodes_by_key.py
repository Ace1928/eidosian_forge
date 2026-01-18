import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _update_nodes_by_key(self, key, value, node_refs):
    """Update the _nodes_by_key dict with a new key.

        For a key of (foo, bar, baz) create
        _nodes_by_key[foo][bar][baz] = key_value
        """
    if self._nodes_by_key is None:
        return
    key_dict = self._nodes_by_key
    if self.reference_lists:
        key_value = StaticTuple(key, value, node_refs)
    else:
        key_value = StaticTuple(key, value)
    for subkey in key[:-1]:
        key_dict = key_dict.setdefault(subkey, {})
    key_dict[key[-1]] = key_value