from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
class _LeafNode(dict):
    """A leaf node for a serialised B+Tree index."""
    __slots__ = ('min_key', 'max_key', '_keys')

    def __init__(self, bytes, key_length, ref_list_length):
        """Parse bytes to create a leaf node object."""
        key_list = _btree_serializer._parse_leaf_lines(bytes, key_length, ref_list_length)
        if key_list:
            self.min_key = key_list[0][0]
            self.max_key = key_list[-1][0]
        else:
            self.min_key = self.max_key = None
        super().__init__(key_list)
        self._keys = dict(self)

    def all_items(self):
        """Return a sorted list of (key, (value, refs)) items"""
        items = sorted(self.items())
        return items

    def all_keys(self):
        """Return a sorted list of all keys."""
        keys = sorted(self.keys())
        return keys