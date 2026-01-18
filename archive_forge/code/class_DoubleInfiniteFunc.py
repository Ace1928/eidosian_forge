import sys
import copy
import heapq
import collections
import functools
import numpy as np
from scipy._lib._util import MapWrapper, _FunctionWrapper
class DoubleInfiniteFunc:
    """
    Argument transform from (-oo, oo) to (-1, 1)
    """

    def __init__(self, func):
        self._func = func
        self._tmin = sys.float_info.min ** 0.5

    def get_t(self, x):
        s = -1 if x < 0 else 1
        return s / (abs(x) + 1)

    def __call__(self, t):
        if abs(t) < self._tmin:
            return 0.0
        else:
            x = (1 - abs(t)) / t
            f = self._func(x)
            return f / t / t