import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _get_cache():
    """Get the per-thread page cache.

    We need a function to do this because in a new thread the _thread_caches
    threading.local object does not have the cache initialized yet.
    """
    page_cache = getattr(_thread_caches, 'page_cache', None)
    if page_cache is None:
        page_cache = lru_cache.LRUSizeCache(_PAGE_CACHE_SIZE)
        _thread_caches.page_cache = page_cache
    return page_cache