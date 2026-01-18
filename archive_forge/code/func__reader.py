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
def _reader(self):
    q = Queue()

    def callback(*args, **kwargs):
        if self._aborted:
            raise AbortThread('canceled by user')
        q.put((args, kwargs))
    self._future = self._executor.submit(self._func, **{self._callback_kwd: callback})
    while True:
        try:
            item = q.get(timeout=self._wait_seconds)
        except Empty:
            pass
        else:
            q.task_done()
            yield item
        if self._future.done():
            break
    remaining = []
    while True:
        try:
            item = q.get_nowait()
        except Empty:
            break
        else:
            q.task_done()
            remaining.append(item)
    q.join()
    yield from remaining