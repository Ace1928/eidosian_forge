import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
class TestLeaks:

    class A:
        iters = 20

        def bound(self, *args):
            return 0

        @staticmethod
        def unbound(*args):
            return 0

    @pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
    @pytest.mark.parametrize('name, incr', [('bound', A.iters), ('unbound', 0)])
    def test_frompyfunc_leaks(self, name, incr):
        import gc
        A_func = getattr(self.A, name)
        gc.disable()
        try:
            refcount = sys.getrefcount(A_func)
            for i in range(self.A.iters):
                a = self.A()
                a.f = np.frompyfunc(getattr(a, name), 1, 1)
                out = a.f(np.arange(10))
            a = None
            assert_equal(sys.getrefcount(A_func), refcount + incr)
            for i in range(5):
                gc.collect()
            assert_equal(sys.getrefcount(A_func), refcount)
        finally:
            gc.enable()