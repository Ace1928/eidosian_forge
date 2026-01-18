import math
import numbers
import random
import sys
from fractions import Fraction
from decimal import Decimal
from itertools import groupby, repeat
from bisect import bisect_left, bisect_right
from math import hypot, sqrt, fabs, exp, erf, tau, log, fsum
from functools import reduce
from operator import mul
from collections import Counter, namedtuple, defaultdict
def _ss(data, c=None):
    """Return the exact mean and sum of square deviations of sequence data.

    Calculations are done in a single pass, allowing the input to be an iterator.

    If given *c* is used the mean; otherwise, it is calculated from the data.
    Use the *c* argument with care, as it can lead to garbage results.

    """
    if c is not None:
        T, ssd, count = _sum(((d := (x - c)) * d for x in data))
        return (T, ssd, c, count)
    count = 0
    types = set()
    types_add = types.add
    sx_partials = defaultdict(int)
    sxx_partials = defaultdict(int)
    for typ, values in groupby(data, type):
        types_add(typ)
        for n, d in map(_exact_ratio, values):
            count += 1
            sx_partials[d] += n
            sxx_partials[d] += n * n
    if not count:
        ssd = c = Fraction(0)
    elif None in sx_partials:
        ssd = c = sx_partials[None]
        assert not _isfinite(ssd)
    else:
        sx = sum((Fraction(n, d) for d, n in sx_partials.items()))
        sxx = sum((Fraction(n, d * d) for d, n in sxx_partials.items()))
        ssd = (count * sxx - sx * sx) / count
        c = sx / count
    T = reduce(_coerce, types, int)
    return (T, ssd, c, count)