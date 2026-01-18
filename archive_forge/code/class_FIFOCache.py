from collections import deque
from typing import Any, Callable, Deque, Dict
class FIFOCache(dict):
    """A class which manages a cache of entries, removing old ones."""

    def __init__(self, max_cache: int=100, after_cleanup_count=None) -> None:
        dict.__init__(self)
        self._max_cache = max_cache
        if after_cleanup_count is None:
            self._after_cleanup_count = self._max_cache * 8 // 10
        else:
            self._after_cleanup_count = min(after_cleanup_count, self._max_cache)
        self._cleanup: Dict[Any, Callable[[], None]] = {}
        self._queue: Deque[Any] = deque()

    def __setitem__(self, key, value):
        """Add a value to the cache, there will be no cleanup function."""
        self.add(key, value, cleanup=None)

    def __delitem__(self, key):
        self._queue.remove(key)
        self._remove(key)

    def add(self, key, value, cleanup=None):
        """Add a new value to the cache.

        Also, if the entry is ever removed from the queue, call cleanup.
        Passing it the key and value being removed.

        :param key: The key to store it under
        :param value: The object to store
        :param cleanup: None or a function taking (key, value) to indicate
                        'value' should be cleaned up
        """
        if key in self:
            del self[key]
        self._queue.append(key)
        dict.__setitem__(self, key, value)
        if cleanup is not None:
            self._cleanup[key] = cleanup
        if len(self) > self._max_cache:
            self.cleanup()

    def cache_size(self):
        """Get the number of entries we will cache."""
        return self._max_cache

    def cleanup(self):
        """Clear the cache until it shrinks to the requested size.

        This does not completely wipe the cache, just makes sure it is under
        the after_cleanup_count.
        """
        while len(self) > self._after_cleanup_count:
            self._remove_oldest()
        if len(self._queue) != len(self):
            raise AssertionError('The length of the queue should always equal the length of the dict. %s != %s' % (len(self._queue), len(self)))

    def clear(self):
        """Clear out all of the cache."""
        while self:
            self._remove_oldest()

    def _remove(self, key):
        """Remove an entry, making sure to call any cleanup function."""
        cleanup = self._cleanup.pop(key, None)
        val = dict.pop(self, key)
        if cleanup is not None:
            cleanup(key, val)
        return val

    def _remove_oldest(self):
        """Remove the oldest entry."""
        key = self._queue.popleft()
        self._remove(key)

    def resize(self, max_cache, after_cleanup_count=None):
        """Increase/decrease the number of cached entries.

        :param max_cache: The maximum number of entries to cache.
        :param after_cleanup_count: After cleanup, we should have at most this
            many entries. This defaults to 80% of max_cache.
        """
        self._max_cache = max_cache
        if after_cleanup_count is None:
            self._after_cleanup_count = max_cache * 8 // 10
        else:
            self._after_cleanup_count = min(max_cache, after_cleanup_count)
        if len(self) > self._max_cache:
            self.cleanup()

    def copy(self):
        raise NotImplementedError(self.copy)

    def pop(self, key, default=None):
        raise NotImplementedError(self.pop)

    def popitem(self):
        raise NotImplementedError(self.popitem)

    def setdefault(self, key, defaultval=None):
        """similar to dict.setdefault"""
        if key in self:
            return self[key]
        self[key] = defaultval
        return defaultval

    def update(self, *args, **kwargs):
        """Similar to dict.update()"""
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, dict):
                for key in arg:
                    self.add(key, arg[key])
            else:
                for key, val in args[0]:
                    self.add(key, val)
        elif len(args) > 1:
            raise TypeError('update expected at most 1 argument, got %d' % len(args))
        if kwargs:
            for key in kwargs:
                self.add(key, kwargs[key])