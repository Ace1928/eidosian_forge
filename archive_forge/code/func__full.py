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
def _full(A):
    while True:
        yield tuple(A)
        for i in range(size - 2, -1, -1):
            if A[i] < A[i + 1]:
                break
        else:
            return
        for j in range(size - 1, i, -1):
            if A[i] < A[j]:
                break
        A[i], A[j] = (A[j], A[i])
        A[i + 1:] = A[:i - size:-1]