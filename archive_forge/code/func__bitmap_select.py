import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def _bitmap_select(s, seq):
    """Select elements of ``seq`` which are marked by the bitmap set ``s``.

    E.g.:

        >>> list(_bitmap_select(0b11010, ['A', 'B', 'C', 'D', 'E']))
        ['B', 'D', 'E']
    """
    return (x for x, b in zip(seq, bin(s)[:1:-1]) if b == '1')