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
class LimitedSet:
    """Kind-of Set (or priority queue) with limitations.

    Good for when you need to test for membership (`a in set`),
    but the set should not grow unbounded.

    ``maxlen`` is enforced at all times, so if the limit is reached
    we'll also remove non-expired items.

    You can also configure ``minlen``: this is the minimal residual size
    of the set.

    All arguments are optional, and no limits are enabled by default.

    Arguments:
        maxlen (int): Optional max number of items.
            Adding more items than ``maxlen`` will result in immediate
            removal of items sorted by oldest insertion time.

        expires (float): TTL for all items.
            Expired items are purged as keys are inserted.

        minlen (int): Minimal residual size of this set.
            .. versionadded:: 4.0

            Value must be less than ``maxlen`` if both are configured.

            Older expired items will be deleted, only after the set
            exceeds ``minlen`` number of items.

        data (Sequence): Initial data to initialize set with.
            Can be an iterable of ``(key, value)`` pairs,
            a dict (``{key: insertion_time}``), or another instance
            of :class:`LimitedSet`.

    Example:
        >>> s = LimitedSet(maxlen=50000, expires=3600, minlen=4000)
        >>> for i in range(60000):
        ...     s.add(i)
        ...     s.add(str(i))
        ...
        >>> 57000 in s  # last 50k inserted values are kept
        True
        >>> '10' in s  # '10' did expire and was purged from set.
        False
        >>> len(s)  # maxlen is reached
        50000
        >>> s.purge(now=time.monotonic() + 7200)  # clock + 2 hours
        >>> len(s)  # now only minlen items are cached
        4000
        >>>> 57000 in s  # even this item is gone now
        False
    """
    max_heap_percent_overload = 15

    def __init__(self, maxlen=0, expires=0, data=None, minlen=0):
        self.maxlen = 0 if maxlen is None else maxlen
        self.minlen = 0 if minlen is None else minlen
        self.expires = 0 if expires is None else expires
        self._data = {}
        self._heap = []
        if data:
            self.update(data)
        if not self.maxlen >= self.minlen >= 0:
            raise ValueError('minlen must be a positive number, less or equal to maxlen.')
        if self.expires < 0:
            raise ValueError('expires cannot be negative!')

    def _refresh_heap(self):
        """Time consuming recreating of heap.  Don't run this too often."""
        self._heap[:] = [entry for entry in self._data.values()]
        heapify(self._heap)

    def _maybe_refresh_heap(self):
        if self._heap_overload >= self.max_heap_percent_overload:
            self._refresh_heap()

    def clear(self):
        """Clear all data, start from scratch again."""
        self._data.clear()
        self._heap[:] = []

    def add(self, item, now=None):
        """Add a new item, or reset the expiry time of an existing item."""
        now = now or time.monotonic()
        if item in self._data:
            self.discard(item)
        entry = (now, item)
        self._data[item] = entry
        heappush(self._heap, entry)
        if self.maxlen and len(self._data) >= self.maxlen:
            self.purge()

    def update(self, other):
        """Update this set from other LimitedSet, dict or iterable."""
        if not other:
            return
        if isinstance(other, LimitedSet):
            self._data.update(other._data)
            self._refresh_heap()
            self.purge()
        elif isinstance(other, dict):
            for key, inserted in other.items():
                if isinstance(inserted, (tuple, list)):
                    inserted = inserted[0]
                if not isinstance(inserted, float):
                    raise ValueError(f'Expecting float timestamp, got type {type(inserted)!r} with value: {inserted}')
                self.add(key, inserted)
        else:
            for obj in other:
                self.add(obj)

    def discard(self, item):
        self._data.pop(item, None)
        self._maybe_refresh_heap()
    pop_value = discard

    def purge(self, now=None):
        """Check oldest items and remove them if needed.

        Arguments:
            now (float): Time of purging -- by default right now.
                This can be useful for unit testing.
        """
        now = now or time.monotonic()
        now = now() if isinstance(now, Callable) else now
        if self.maxlen:
            while len(self._data) > self.maxlen:
                self.pop()
        if self.expires:
            while len(self._data) > self.minlen >= 0:
                inserted_time, _ = self._heap[0]
                if inserted_time + self.expires > now:
                    break
                self.pop()

    def pop(self, default: Any=None) -> Any:
        """Remove and return the oldest item, or :const:`None` when empty."""
        while self._heap:
            _, item = heappop(self._heap)
            try:
                self._data.pop(item)
            except KeyError:
                pass
            else:
                return item
        return default

    def as_dict(self):
        """Whole set as serializable dictionary.

        Example:
            >>> s = LimitedSet(maxlen=200)
            >>> r = LimitedSet(maxlen=200)
            >>> for i in range(500):
            ...     s.add(i)
            ...
            >>> r.update(s.as_dict())
            >>> r == s
            True
        """
        return {key: inserted for inserted, key in self._data.values()}

    def __eq__(self, other):
        return self._data == other._data

    def __repr__(self):
        return REPR_LIMITED_SET.format(self, name=type(self).__name__, size=len(self))

    def __iter__(self):
        return (i for _, i in sorted(self._data.values()))

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data

    def __reduce__(self):
        return (self.__class__, (self.maxlen, self.expires, self.as_dict(), self.minlen))

    def __bool__(self):
        return bool(self._data)
    __nonzero__ = __bool__

    @property
    def _heap_overload(self):
        """Compute how much is heap bigger than data [percents]."""
        return len(self._heap) * 100 / max(len(self._data), 1) - 100