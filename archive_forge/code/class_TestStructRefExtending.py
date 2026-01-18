import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
class TestStructRefExtending(MemoryLeakMixin, TestCase):

    def test_overload_method(self):

        @njit
        def check(x):
            vs = np.arange(10, dtype=np.float64)
            ctr = 11
            obj = MyStruct(vs, ctr)
            return obj.testme(x)
        x = 3
        got = check(x)
        expect = check.py_func(x)
        self.assertPreciseEqual(got, expect)

    def test_overload_attribute(self):

        @njit
        def check():
            vs = np.arange(10, dtype=np.float64)
            ctr = 11
            obj = MyStruct(vs, ctr)
            return obj.prop
        got = check()
        expect = check.py_func()
        self.assertPreciseEqual(got, expect)