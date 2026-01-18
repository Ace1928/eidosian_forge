import warnings
from collections import Counter, defaultdict, deque, abc
from collections.abc import Sequence
from functools import partial, reduce, wraps
from heapq import heapify, heapreplace, heappop
from itertools import (
from math import exp, factorial, floor, log
from queue import Empty, Queue
from random import random, randrange, uniform
from operator import itemgetter, mul, sub, gt, lt, ge, le
from sys import hexversion, maxsize
from time import monotonic
from .recipes import (
class seekable:
    """Wrap an iterator to allow for seeking backward and forward. This
    progressively caches the items in the source iterable so they can be
    re-visited.

    Call :meth:`seek` with an index to seek to that position in the source
    iterable.

    To "reset" an iterator, seek to ``0``:

        >>> from itertools import count
        >>> it = seekable((str(n) for n in count()))
        >>> next(it), next(it), next(it)
        ('0', '1', '2')
        >>> it.seek(0)
        >>> next(it), next(it), next(it)
        ('0', '1', '2')
        >>> next(it)
        '3'

    You can also seek forward:

        >>> it = seekable((str(n) for n in range(20)))
        >>> it.seek(10)
        >>> next(it)
        '10'
        >>> it.seek(20)  # Seeking past the end of the source isn't a problem
        >>> list(it)
        []
        >>> it.seek(0)  # Resetting works even after hitting the end
        >>> next(it), next(it), next(it)
        ('0', '1', '2')

    Call :meth:`peek` to look ahead one item without advancing the iterator:

        >>> it = seekable('1234')
        >>> it.peek()
        '1'
        >>> list(it)
        ['1', '2', '3', '4']
        >>> it.peek(default='empty')
        'empty'

    Before the iterator is at its end, calling :func:`bool` on it will return
    ``True``. After it will return ``False``:

        >>> it = seekable('5678')
        >>> bool(it)
        True
        >>> list(it)
        ['5', '6', '7', '8']
        >>> bool(it)
        False

    You may view the contents of the cache with the :meth:`elements` method.
    That returns a :class:`SequenceView`, a view that updates automatically:

        >>> it = seekable((str(n) for n in range(10)))
        >>> next(it), next(it), next(it)
        ('0', '1', '2')
        >>> elements = it.elements()
        >>> elements
        SequenceView(['0', '1', '2'])
        >>> next(it)
        '3'
        >>> elements
        SequenceView(['0', '1', '2', '3'])

    By default, the cache grows as the source iterable progresses, so beware of
    wrapping very large or infinite iterables. Supply *maxlen* to limit the
    size of the cache (this of course limits how far back you can seek).

        >>> from itertools import count
        >>> it = seekable((str(n) for n in count()), maxlen=2)
        >>> next(it), next(it), next(it), next(it)
        ('0', '1', '2', '3')
        >>> list(it.elements())
        ['2', '3']
        >>> it.seek(0)
        >>> next(it), next(it), next(it), next(it)
        ('2', '3', '4', '5')
        >>> next(it)
        '6'

    """

    def __init__(self, iterable, maxlen=None):
        self._source = iter(iterable)
        if maxlen is None:
            self._cache = []
        else:
            self._cache = deque([], maxlen)
        self._index = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._index is not None:
            try:
                item = self._cache[self._index]
            except IndexError:
                self._index = None
            else:
                self._index += 1
                return item
        item = next(self._source)
        self._cache.append(item)
        return item

    def __bool__(self):
        try:
            self.peek()
        except StopIteration:
            return False
        return True

    def peek(self, default=_marker):
        try:
            peeked = next(self)
        except StopIteration:
            if default is _marker:
                raise
            return default
        if self._index is None:
            self._index = len(self._cache)
        self._index -= 1
        return peeked

    def elements(self):
        return SequenceView(self._cache)

    def seek(self, index):
        self._index = index
        remainder = index - len(self._cache)
        if remainder > 0:
            consume(self, remainder)