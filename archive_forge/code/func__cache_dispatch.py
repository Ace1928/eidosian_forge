import collections
from typing import Optional, Iterable
from tensorflow.core.function.polymorphism import function_type
def _cache_dispatch(self, request, target):
    """Caches the dispatch lookup result for a target."""
    if target is not None:
        if len(self._dispatch_cache) > _MAX_DISPATCH_CACHE:
            self._dispatch_cache.popitem(last=False)
        self._dispatch_cache[request] = target