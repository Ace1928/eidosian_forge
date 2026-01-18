from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
class FakeLruCache:
    """Doesn't actually cache but supports LRU interface in Python 2."""

    def __init__(self, function):
        self._function = function

    def cache_clear(self):
        """Exposes this function of actual LRU to avoid missing attribute errors."""
        pass

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)