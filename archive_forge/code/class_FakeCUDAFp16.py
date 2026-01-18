from contextlib import contextmanager
import sys
import threading
import traceback
from numba.core import types
import numpy as np
from numba.np import numpy_support
from .vector_types import vector_types
class FakeCUDAFp16(object):

    def hadd(self, a, b):
        return a + b

    def hsub(self, a, b):
        return a - b

    def hmul(self, a, b):
        return a * b

    def hdiv(self, a, b):
        return a / b

    def hfma(self, a, b, c):
        return a * b + c

    def hneg(self, a):
        return -a

    def habs(self, a):
        return abs(a)

    def hsin(self, x):
        return np.sin(x, dtype=np.float16)

    def hcos(self, x):
        return np.cos(x, dtype=np.float16)

    def hlog(self, x):
        return np.log(x, dtype=np.float16)

    def hlog2(self, x):
        return np.log2(x, dtype=np.float16)

    def hlog10(self, x):
        return np.log10(x, dtype=np.float16)

    def hexp(self, x):
        return np.exp(x, dtype=np.float16)

    def hexp2(self, x):
        return np.exp2(x, dtype=np.float16)

    def hexp10(self, x):
        return np.float16(10 ** x)

    def hsqrt(self, x):
        return np.sqrt(x, dtype=np.float16)

    def hrsqrt(self, x):
        return np.float16(x ** (-0.5))

    def hceil(self, x):
        return np.ceil(x, dtype=np.float16)

    def hfloor(self, x):
        return np.ceil(x, dtype=np.float16)

    def hrcp(self, x):
        return np.reciprocal(x, dtype=np.float16)

    def htrunc(self, x):
        return np.trunc(x, dtype=np.float16)

    def hrint(self, x):
        return np.rint(x, dtype=np.float16)

    def heq(self, a, b):
        return a == b

    def hne(self, a, b):
        return a != b

    def hge(self, a, b):
        return a >= b

    def hgt(self, a, b):
        return a > b

    def hle(self, a, b):
        return a <= b

    def hlt(self, a, b):
        return a < b

    def hmax(self, a, b):
        return max(a, b)

    def hmin(self, a, b):
        return min(a, b)