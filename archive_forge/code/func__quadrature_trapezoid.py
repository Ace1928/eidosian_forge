import sys
import copy
import heapq
import collections
import functools
import numpy as np
from scipy._lib._util import MapWrapper, _FunctionWrapper
def _quadrature_trapezoid(x1, x2, f, norm_func):
    """
    Composite trapezoid quadrature
    """
    x3 = 0.5 * (x1 + x2)
    f1 = f(x1)
    f2 = f(x2)
    f3 = f(x3)
    s2 = 0.25 * (x2 - x1) * (f1 + 2 * f3 + f2)
    round_err = 0.25 * abs(x2 - x1) * (float(norm_func(f1)) + 2 * float(norm_func(f3)) + float(norm_func(f2))) * 2e-16
    s1 = 0.5 * (x2 - x1) * (f1 + f2)
    err = 1 / 3 * float(norm_func(s1 - s2))
    return (s2, err, round_err)