import sys
import copy
import heapq
import collections
import functools
import numpy as np
from scipy._lib._util import MapWrapper, _FunctionWrapper
def _subdivide_interval(args):
    interval, f, norm_func, _quadrature = args
    old_err, a, b, old_int = interval
    c = 0.5 * (a + b)
    if getattr(_quadrature, 'cache_size', 0) > 0:
        f = functools.lru_cache(_quadrature.cache_size)(f)
    s1, err1, round1 = _quadrature(a, c, f, norm_func)
    dneval = _quadrature.num_eval
    s2, err2, round2 = _quadrature(c, b, f, norm_func)
    dneval += _quadrature.num_eval
    if old_int is None:
        old_int, _, _ = _quadrature(a, b, f, norm_func)
        dneval += _quadrature.num_eval
    if getattr(_quadrature, 'cache_size', 0) > 0:
        dneval = f.cache_info().misses
    dint = s1 + s2 - old_int
    derr = err1 + err2 - old_err
    dround_err = round1 + round2
    subintervals = ((a, c, s1, err1), (c, b, s2, err2))
    return (dint, derr, dround_err, subintervals, dneval)