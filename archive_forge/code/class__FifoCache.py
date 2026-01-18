import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import collections
import pprint
import traceback
import types
from datetime import datetime
class _FifoCache(object):

    def __init__(self, size):
        self.not_in_cache = not_in_cache = object()
        cache = {}
        key_fifo = collections.deque([], size)

        def get(self, key):
            return cache.get(key, not_in_cache)

        def set(self, key, value):
            cache[key] = value
            while len(key_fifo) > size:
                cache.pop(key_fifo.popleft(), None)
            key_fifo.append(key)

        def clear(self):
            cache.clear()
            key_fifo.clear()

        def cache_len(self):
            return len(cache)
        self.get = types.MethodType(get, self)
        self.set = types.MethodType(set, self)
        self.clear = types.MethodType(clear, self)
        self.__len__ = types.MethodType(cache_len, self)