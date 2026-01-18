from collections import deque
from typing import Any, Callable, Deque, Dict
class FIFOSizeCache(FIFOCache):
    """An FIFOCache that removes things based on the size of the values.

    This differs in that it doesn't care how many actual items there are,
    it restricts the cache to be cleaned based on the size of the data.
    """

    def __init__(self, max_size=1024 * 1024, after_cleanup_size=None, compute_size=None):
        """Create a new FIFOSizeCache.

        :param max_size: The max number of bytes to store before we start
            clearing out entries.
        :param after_cleanup_size: After cleaning up, shrink everything to this
            size (defaults to 80% of max_size).
        :param compute_size: A function to compute the size of a value. If
            not supplied we default to 'len'.
        """
        FIFOCache.__init__(self, max_cache=max_size)
        self._max_size = max_size
        if after_cleanup_size is None:
            self._after_cleanup_size = self._max_size * 8 // 10
        else:
            self._after_cleanup_size = min(after_cleanup_size, self._max_size)
        self._value_size = 0
        self._compute_size = compute_size
        if compute_size is None:
            self._compute_size = len

    def add(self, key, value, cleanup=None):
        """Add a new value to the cache.

        Also, if the entry is ever removed from the queue, call cleanup.
        Passing it the key and value being removed.

        :param key: The key to store it under
        :param value: The object to store, this value by itself is >=
            after_cleanup_size, then we will not store it at all.
        :param cleanup: None or a function taking (key, value) to indicate
                        'value' sohuld be cleaned up.
        """
        if key in self:
            del self[key]
        value_len = self._compute_size(value)
        if value_len >= self._after_cleanup_size:
            return
        self._queue.append(key)
        dict.__setitem__(self, key, value)
        if cleanup is not None:
            self._cleanup[key] = cleanup
        self._value_size += value_len
        if self._value_size > self._max_size:
            self.cleanup()

    def cache_size(self):
        """Get the number of bytes we will cache."""
        return self._max_size

    def cleanup(self):
        """Clear the cache until it shrinks to the requested size.

        This does not completely wipe the cache, just makes sure it is under
        the after_cleanup_size.
        """
        while self._value_size > self._after_cleanup_size:
            self._remove_oldest()

    def _remove(self, key):
        """Remove an entry, making sure to maintain the invariants."""
        val = FIFOCache._remove(self, key)
        self._value_size -= self._compute_size(val)
        return val

    def resize(self, max_size, after_cleanup_size=None):
        """Increase/decrease the amount of cached data.

        :param max_size: The maximum number of bytes to cache.
        :param after_cleanup_size: After cleanup, we should have at most this
            many bytes cached. This defaults to 80% of max_size.
        """
        FIFOCache.resize(self, max_size)
        self._max_size = max_size
        if after_cleanup_size is None:
            self._after_cleanup_size = max_size * 8 // 10
        else:
            self._after_cleanup_size = min(max_size, after_cleanup_size)
        if self._value_size > self._max_size:
            self.cleanup()