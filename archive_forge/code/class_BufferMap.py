import time
from collections import OrderedDict as _OrderedDict
from collections import deque
from collections.abc import Callable, Mapping, MutableMapping, MutableSet, Sequence
from heapq import heapify, heappop, heappush
from itertools import chain, count
from queue import Empty
from typing import Any, Dict, Iterable, List  # noqa
from .functional import first, uniq
from .text import match_case
class BufferMap(OrderedDict, Evictable):
    """Map of buffers."""
    Buffer = Messagebuffer
    Empty = Empty
    maxsize = None
    total = 0
    bufmaxsize = None

    def __init__(self, maxsize, iterable=None, bufmaxsize=1000):
        super().__init__()
        self.maxsize = maxsize
        self.bufmaxsize = 1000
        if iterable:
            self.update(iterable)
        self.total = sum((len(buf) for buf in self.items()))

    def put(self, key, item):
        self._get_or_create_buffer(key).put(item)
        self.total += 1
        self.move_to_end(key)
        self.maxsize and self._evict()

    def extend(self, key, it):
        self._get_or_create_buffer(key).extend(it)
        self.total += len(it)
        self.maxsize and self._evict()

    def take(self, key, *default):
        item, throw = (None, False)
        try:
            buf = self[key]
        except KeyError:
            throw = True
        else:
            try:
                item = buf.take()
                self.total -= 1
            except self.Empty:
                throw = True
            else:
                self.move_to_end(key)
        if throw:
            if default:
                return default[0]
            raise self.Empty()
        return item

    def _get_or_create_buffer(self, key):
        try:
            return self[key]
        except KeyError:
            buf = self[key] = self._new_buffer()
            return buf

    def _new_buffer(self):
        return self.Buffer(maxsize=self.bufmaxsize)

    def _LRUpop(self, *default):
        return self[self._LRUkey()].take(*default)

    def _pop_to_evict(self):
        for _ in range(100):
            key = self._LRUkey()
            buf = self[key]
            try:
                buf.take()
            except (IndexError, self.Empty):
                self.pop(key)
            else:
                self.total -= 1
                if not len(buf):
                    self.pop(key)
                else:
                    self.move_to_end(key)
                break

    def __repr__(self):
        return f'<{type(self).__name__}: {self.total}/{self.maxsize}>'

    @property
    def _evictcount(self):
        return self.total