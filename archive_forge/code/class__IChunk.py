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
class _IChunk:

    def __init__(self, iterable, n):
        self._it = islice(iterable, n)
        self._cache = deque()

    def fill_cache(self):
        self._cache.extend(self._it)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._it)
        except StopIteration:
            if self._cache:
                return self._cache.popleft()
            else:
                raise