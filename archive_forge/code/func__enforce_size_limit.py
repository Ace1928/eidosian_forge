from __future__ import annotations
import threading
from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
from typing import Any, Callable, TypeVar
def _enforce_size_limit(self, capacity: int) -> None:
    """Shrink the cache if necessary, evicting the oldest items."""
    while len(self._cache) > capacity:
        key, value = self._cache.popitem(last=False)
        if self._on_evict is not None:
            self._on_evict(key, value)