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
def _islice_helper(it, s):
    start = s.start
    stop = s.stop
    if s.step == 0:
        raise ValueError('step argument must be a non-zero integer or None.')
    step = s.step or 1
    if step > 0:
        start = 0 if start is None else start
        if start < 0:
            cache = deque(enumerate(it, 1), maxlen=-start)
            len_iter = cache[-1][0] if cache else 0
            i = max(len_iter + start, 0)
            if stop is None:
                j = len_iter
            elif stop >= 0:
                j = min(stop, len_iter)
            else:
                j = max(len_iter + stop, 0)
            n = j - i
            if n <= 0:
                return
            for index, item in islice(cache, 0, n, step):
                yield item
        elif stop is not None and stop < 0:
            next(islice(it, start, start), None)
            cache = deque(islice(it, -stop), maxlen=-stop)
            for index, item in enumerate(it):
                cached_item = cache.popleft()
                if index % step == 0:
                    yield cached_item
                cache.append(item)
        else:
            yield from islice(it, start, stop, step)
    else:
        start = -1 if start is None else start
        if stop is not None and stop < 0:
            n = -stop - 1
            cache = deque(enumerate(it, 1), maxlen=n)
            len_iter = cache[-1][0] if cache else 0
            if start < 0:
                i, j = (start, stop)
            else:
                i, j = (min(start - len_iter, -1), None)
            for index, item in list(cache)[i:j:step]:
                yield item
        else:
            if stop is not None:
                m = stop + 1
                next(islice(it, m, m), None)
            if start < 0:
                i = start
                n = None
            elif stop is None:
                i = None
                n = start + 1
            else:
                i = None
                n = start - stop
                if n <= 0:
                    return
            cache = list(islice(it, n))
            yield from cache[i::step]