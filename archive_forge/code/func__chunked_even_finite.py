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
def _chunked_even_finite(iterable, N, n):
    if N < 1:
        return
    q, r = divmod(N, n)
    num_lists = q + (1 if r > 0 else 0)
    q, r = divmod(N, num_lists)
    full_size = q + (1 if r > 0 else 0)
    partial_size = full_size - 1
    num_full = N - partial_size * num_lists
    num_partial = num_lists - num_full
    buffer = []
    iterator = iter(iterable)
    for x in iterator:
        buffer.append(x)
        if len(buffer) == full_size:
            yield buffer
            buffer = []
            num_full -= 1
            if num_full <= 0:
                break
    for x in iterator:
        buffer.append(x)
        if len(buffer) == partial_size:
            yield buffer
            buffer = []
            num_partial -= 1