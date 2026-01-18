from __future__ import unicode_literals
from collections import deque
from functools import wraps
class FastDictCache(dict):
    """
    Fast, lightweight cache which keeps at most `size` items.
    It will discard the oldest items in the cache first.

    The cache is a dictionary, which doesn't keep track of access counts.
    It is perfect to cache little immutable objects which are not expensive to
    create, but where a dictionary lookup is still much faster than an object
    instantiation.

    :param get_value: Callable that's called in case of a missing key.
    """

    def __init__(self, get_value=None, size=1000000):
        assert callable(get_value)
        assert isinstance(size, int) and size > 0
        self._keys = deque()
        self.get_value = get_value
        self.size = size

    def __missing__(self, key):
        if len(self) > self.size:
            key_to_remove = self._keys.popleft()
            if key_to_remove in self:
                del self[key_to_remove]
        result = self.get_value(*key)
        self[key] = result
        self._keys.append(key)
        return result