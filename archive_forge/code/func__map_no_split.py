import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _map_no_split(self, key, value):
    """Map a key to a value.

        This assumes either the key does not already exist, or you have already
        removed its size and length from self.

        :return: True if adding this node should cause us to split.
        """
    self._items[key] = value
    self._raw_size += self._key_value_len(key, value)
    self._len += 1
    serialised_key = self._serialise_key(key)
    if self._common_serialised_prefix is None:
        self._common_serialised_prefix = serialised_key
    else:
        self._common_serialised_prefix = self.common_prefix(self._common_serialised_prefix, serialised_key)
    search_key = self._search_key(key)
    if self._search_prefix is _unknown:
        self._compute_search_prefix()
    if self._search_prefix is None:
        self._search_prefix = search_key
    else:
        self._search_prefix = self.common_prefix(self._search_prefix, search_key)
    if self._len > 1 and self._maximum_size and (self._current_size() > self._maximum_size):
        if search_key != self._search_prefix or not self._are_search_keys_identical():
            return True
    return False