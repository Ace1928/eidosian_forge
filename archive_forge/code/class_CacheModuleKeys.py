from __future__ import (absolute_import, division, print_function)
import collections
import os
import time
from multiprocessing import Lock
from itertools import chain
from ansible.errors import AnsibleError
from ansible.module_utils.common._collections_compat import MutableSet
from ansible.plugins.cache import BaseCacheModule
from ansible.utils.display import Display
class CacheModuleKeys(MutableSet):
    """
    A set subclass that keeps track of insertion time and persists
    the set in memcached.
    """
    PREFIX = 'ansible_cache_keys'

    def __init__(self, cache, *args, **kwargs):
        self._cache = cache
        self._keyset = dict(*args, **kwargs)

    def __contains__(self, key):
        return key in self._keyset

    def __iter__(self):
        return iter(self._keyset)

    def __len__(self):
        return len(self._keyset)

    def add(self, value):
        self._keyset[value] = time.time()
        self._cache.set(self.PREFIX, self._keyset)

    def discard(self, value):
        del self._keyset[value]
        self._cache.set(self.PREFIX, self._keyset)

    def remove_by_timerange(self, s_min, s_max):
        for k in list(self._keyset.keys()):
            t = self._keyset[k]
            if s_min < t < s_max:
                del self._keyset[k]
        self._cache.set(self.PREFIX, self._keyset)