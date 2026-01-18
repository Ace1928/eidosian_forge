from __future__ import unicode_literals
from collections import deque
from functools import wraps
class SimpleCache(object):
    """
    Very simple cache that discards the oldest item when the cache size is
    exceeded.

    :param maxsize: Maximum size of the cache. (Don't make it too big.)
    """

    def __init__(self, maxsize=8):
        assert isinstance(maxsize, int) and maxsize > 0
        self._data = {}
        self._keys = deque()
        self.maxsize = maxsize

    def get(self, key, getter_func):
        """
        Get object from the cache.
        If not found, call `getter_func` to resolve it, and put that on the top
        of the cache instead.
        """
        try:
            return self._data[key]
        except KeyError:
            value = getter_func()
            self._data[key] = value
            self._keys.append(key)
            if len(self._data) > self.maxsize:
                key_to_remove = self._keys.popleft()
                if key_to_remove in self._data:
                    del self._data[key_to_remove]
            return value

    def clear(self):
        """ Clear cache. """
        self._data = {}
        self._keys = deque()