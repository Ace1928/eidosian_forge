from abc import abstractmethod
from abc import ABCMeta
import threading
import time
import uuid
class CacheMaker(object):
    """Generates decorators that can be cleared later
    """

    def __init__(self, maxsize=None, timeout=_DEFAULT_TIMEOUT):
        """Create cache decorator factory.

        - maxsize : the default size for created caches.

        - timeout : the defaut expiraiton time for created caches.
        """
        self._maxsize = maxsize
        self._timeout = timeout
        self._cache = {}

    def _resolve_setting(self, name=None, maxsize=None, timeout=None):
        if name is None:
            while True:
                name = str(uuid.uuid4())
                if name not in self._cache:
                    break
        if name in self._cache:
            raise KeyError('cache %s already in use' % name)
        if maxsize is None:
            maxsize = self._maxsize
        if maxsize is None:
            raise ValueError('Cache must have a maxsize set')
        if timeout is None:
            timeout = self._timeout
        return (name, maxsize, timeout)

    def memoized(self, name=None):
        name, maxsize, _ = self._resolve_setting(name, 0)
        cache = self._cache[name] = UnboundedCache()
        return lru_cache(None, cache)

    def lrucache(self, name=None, maxsize=None):
        """Named arguments:
        
        - name (optional) is a string, and should be unique amongst all caches

        - maxsize (optional) is an int, overriding any default value set by
          the constructor
        """
        name, maxsize, _ = self._resolve_setting(name, maxsize)
        cache = self._cache[name] = LRUCache(maxsize)
        return lru_cache(maxsize, cache)

    def expiring_lrucache(self, name=None, maxsize=None, timeout=None):
        'Named arguments:\n\n        - name (optional) is a string, and should be unique amongst all caches\n\n        - maxsize (optional) is an int, overriding any default value set by\n          the constructor\n\n        - timeout (optional) is an int, overriding any default value set by\n          the constructor or the default value (%d seconds)\n        ' % _DEFAULT_TIMEOUT
        name, maxsize, timeout = self._resolve_setting(name, maxsize, timeout)
        cache = self._cache[name] = ExpiringLRUCache(maxsize, timeout)
        return lru_cache(maxsize, cache, timeout)

    def clear(self, *names):
        """Clear the given cache(s).
        
        If no 'names' are passed, clear all caches.
        """
        if len(names) == 0:
            names = self._cache.keys()
        for name in names:
            self._cache[name].clear()