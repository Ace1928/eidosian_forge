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
def _invalidate_cache(self):
    if self._cache is not None:
        self._cache = []
        self._cache_complete = False
        self._cache_gen = self._iter()
        if self._cache_lock.locked():
            self._cache_lock.release()
    self._len = None