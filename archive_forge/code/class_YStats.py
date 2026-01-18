import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
class YStats:
    """
    Main Stats class where we collect the information from _yappi and apply the user filters.
    """

    def __init__(self):
        self._clock_type = None
        self._as_dict = {}
        self._as_list = []

    def get(self):
        self._clock_type = _yappi.get_clock_type()
        self.sort(DEFAULT_SORT_TYPE, DEFAULT_SORT_ORDER)
        return self

    def sort(self, sort_type, sort_order):
        self._as_list.sort(key=lambda stat: stat[sort_type].lower() if isinstance(stat[sort_type], str) else stat[sort_type], reverse=sort_order == SORT_ORDERS['desc'])
        return self

    def clear(self):
        del self._as_list[:]
        self._as_dict.clear()

    def empty(self):
        return len(self._as_list) == 0

    def __getitem__(self, key):
        try:
            return self._as_list[key]
        except IndexError:
            return None

    def count(self, item):
        return self._as_list.count(item)

    def __iter__(self):
        return iter(self._as_list)

    def __len__(self):
        return len(self._as_list)

    def pop(self):
        item = self._as_list.pop()
        del self._as_dict[item]
        return item

    def append(self, item):
        existing = self._as_dict.get(item)
        if existing:
            existing += item
            return
        self._as_list.append(item)
        self._as_dict[item] = item

    def _print_header(self, out, columns):
        for x in sorted(columns.keys()):
            title, size = columns[x]
            if len(title) > size:
                raise YappiError('Column title exceeds available length[%s:%d]' % (title, size))
            out.write(title)
            out.write(' ' * (COLUMN_GAP + size - len(title)))
        out.write(LINESEP)

    def _debug_check_sanity(self):
        """
        Check for basic sanity errors in stats. e.g: Check for duplicate stats.
        """
        for x in self:
            if self.count(x) > 1:
                return False
        return True