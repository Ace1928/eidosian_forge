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
def _init_len(self):
    if self._growing:
        start = self._start
        stop = self._stop
        step = self._step
    else:
        start = self._stop
        stop = self._start
        step = -self._step
    distance = stop - start
    if distance <= self._zero:
        self._len = 0
    else:
        q, r = divmod(distance, step)
        self._len = int(q) + int(r != self._zero)