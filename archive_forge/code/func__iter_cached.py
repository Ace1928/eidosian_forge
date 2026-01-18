import calendar
import datetime
import heapq
import itertools
import re
import sys
from functools import wraps
from warnings import warn
from six import advance_iterator, integer_types
from six.moves import _thread, range
from ._common import weekday as weekdaybase
def _iter_cached(self):
    i = 0
    gen = self._cache_gen
    cache = self._cache
    acquire = self._cache_lock.acquire
    release = self._cache_lock.release
    while gen:
        if i == len(cache):
            acquire()
            if self._cache_complete:
                break
            try:
                for j in range(10):
                    cache.append(advance_iterator(gen))
            except StopIteration:
                self._cache_gen = gen = None
                self._cache_complete = True
                break
            release()
        yield cache[i]
        i += 1
    while i < self._len:
        yield cache[i]
        i += 1