import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _search_key_plain(key):
    """Map the key tuple into a search string that just uses the key bytes."""
    return b'\x00'.join(key)