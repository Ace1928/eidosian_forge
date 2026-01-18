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
def _partial(A, r):
    head, tail = (A[:r], A[r:])
    right_head_indexes = range(r - 1, -1, -1)
    left_tail_indexes = range(len(tail))
    while True:
        yield tuple(head)
        pivot = tail[-1]
        for i in right_head_indexes:
            if head[i] < pivot:
                break
            pivot = head[i]
        else:
            return
        for j in left_tail_indexes:
            if tail[j] > head[i]:
                head[i], tail[j] = (tail[j], head[i])
                break
        else:
            for j in right_head_indexes:
                if head[j] > head[i]:
                    head[i], head[j] = (head[j], head[i])
                    break
        tail += head[:i - r:-1]
        i += 1
        head[i:], tail[:] = (tail[:r - i], tail[r - i:])