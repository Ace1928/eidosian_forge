import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def _tree_to_sequence(c):
    """
    Converts a contraction tree to a contraction path as it has to be
    returned by path optimizers. A contraction tree can either be an int
    (=no contraction) or a tuple containing the terms to be contracted. An
    arbitrary number (>= 1) of terms can be contracted at once. Note that
    contractions are commutative, e.g. (j, k, l) = (k, l, j). Note that in
    general, solutions are not unique.

    Parameters
    ----------
    c : tuple or int
        Contraction tree

    Returns
    -------
    path : list[set[int]]
        Contraction path

    Examples
    --------
    >>> _tree_to_sequence(((1,2),(0,(4,5,3))))
    [(1, 2), (1, 2, 3), (0, 2), (0, 1)]
    """
    if type(c) == int:
        return []
    c = [c]
    t = []
    s = []
    while len(c) > 0:
        j = c.pop(-1)
        s.insert(0, tuple())
        for i in sorted([i for i in j if type(i) == int]):
            s[0] += (sum((1 for q in t if q < i)),)
            t.insert(s[0][-1], i)
        for i in [i for i in j if type(i) != int]:
            s[0] += (len(t) + len(c),)
            c.append(i)
    return s