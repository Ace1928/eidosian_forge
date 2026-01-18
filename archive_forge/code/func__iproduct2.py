from collections import Counter, defaultdict, OrderedDict
from itertools import (
from itertools import product as cartes # noqa: F401
from operator import gt
from sympy.utilities.enumerative import (
from sympy.utilities.misc import as_int
from sympy.utilities.decorator import deprecated
def _iproduct2(iterable1, iterable2):
    """Cartesian product of two possibly infinite iterables"""
    it1 = iter(iterable1)
    it2 = iter(iterable2)
    elems1 = []
    elems2 = []
    sentinel = object()

    def append(it, elems):
        e = next(it, sentinel)
        if e is not sentinel:
            elems.append(e)
    n = 0
    append(it1, elems1)
    append(it2, elems2)
    while n <= len(elems1) + len(elems2):
        for m in range(n - len(elems1) + 1, len(elems2)):
            yield (elems1[n - m], elems2[m])
        n += 1
        append(it1, elems1)
        append(it2, elems2)