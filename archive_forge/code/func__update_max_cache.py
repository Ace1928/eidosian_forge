from typing import Callable, Dict, Generic, Iterable, Iterator, Optional, TypeVar
def _update_max_cache(self, max_cache, after_cleanup_count=None):
    self._max_cache = max_cache
    if after_cleanup_count is None:
        self._after_cleanup_count = self._max_cache * 8 / 10
    else:
        self._after_cleanup_count = min(after_cleanup_count, self._max_cache)
    self.cleanup()