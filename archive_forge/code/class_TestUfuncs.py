import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
class TestUfuncs:

    def setup_method(self):
        self.d = (array([1.0, 0, -1, pi / 2] * 2, mask=[0, 1] + [0] * 6), array([1.0, 0, -1, pi / 2] * 2, mask=[1, 0] + [0] * 6))
        self.err_status = np.geterr()
        np.seterr(divide='ignore', invalid='ignore')

    def teardown_method(self):
        np.seterr(**self.err_status)

    def test_testUfuncRegression(self):
        for f in ['sqrt', 'log', 'log10', 'exp', 'conjugate', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh', 'absolute', 'fabs', 'negative', 'floor', 'ceil', 'logical_not', 'add', 'subtract', 'multiply', 'divide', 'true_divide', 'floor_divide', 'remainder', 'fmod', 'hypot', 'arctan2', 'equal', 'not_equal', 'less_equal', 'greater_equal', 'less', 'greater', 'logical_and', 'logical_or', 'logical_xor']:
            try:
                uf = getattr(umath, f)
            except AttributeError:
                uf = getattr(fromnumeric, f)
            mf = getattr(numpy.ma.core, f)
            args = self.d[:uf.nin]
            ur = uf(*args)
            mr = mf(*args)
            assert_equal(ur.filled(0), mr.filled(0), f)
            assert_mask_equal(ur.mask, mr.mask, err_msg=f)

    def test_reduce(self):
        a = self.d[0]
        assert_(not alltrue(a, axis=0))
        assert_(sometrue(a, axis=0))
        assert_equal(sum(a[:3], axis=0), 0)
        assert_equal(product(a, axis=0), 0)
        assert_equal(add.reduce(a), pi)

    def test_minmax(self):
        a = arange(1, 13).reshape(3, 4)
        amask = masked_where(a < 5, a)
        assert_equal(amask.max(), a.max())
        assert_equal(amask.min(), 5)
        assert_equal(amask.max(0), a.max(0))
        assert_equal(amask.min(0), [5, 6, 7, 8])
        assert_(amask.max(1)[0].mask)
        assert_(amask.min(1)[0].mask)

    def test_ndarray_mask(self):
        a = masked_array([-1, 0, 1, 2, 3], mask=[0, 0, 0, 0, 1])
        test = np.sqrt(a)
        control = masked_array([-1, 0, 1, np.sqrt(2), -1], mask=[1, 0, 0, 0, 1])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        assert_(not isinstance(test.mask, MaskedArray))

    def test_treatment_of_NotImplemented(self):
        a = masked_array([1.0, 2.0], mask=[1, 0])
        assert_raises(TypeError, operator.mul, a, 'abc')
        assert_raises(TypeError, operator.truediv, a, 'abc')

        class MyClass:
            __array_priority__ = a.__array_priority__ + 1

            def __mul__(self, other):
                return 'My mul'

            def __rmul__(self, other):
                return 'My rmul'
        me = MyClass()
        assert_(me * a == 'My mul')
        assert_(a * me == 'My rmul')

        class MyClass2:
            __array_priority__ = 100

            def __mul__(self, other):
                return 'Me2mul'

            def __rmul__(self, other):
                return 'Me2rmul'

            def __rdiv__(self, other):
                return 'Me2rdiv'
            __rtruediv__ = __rdiv__
        me_too = MyClass2()
        assert_(a.__mul__(me_too) is NotImplemented)
        assert_(all(multiply.outer(a, me_too) == 'Me2rmul'))
        assert_(a.__truediv__(me_too) is NotImplemented)
        assert_(me_too * a == 'Me2mul')
        assert_(a * me_too == 'Me2rmul')
        assert_(a / me_too == 'Me2rdiv')

    def test_no_masked_nan_warnings(self):
        m = np.ma.array([0.5, np.nan], mask=[0, 1])
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            exp(m)
            add(m, 1)
            m > 0
            sqrt(m)
            log(m)
            tan(m)
            arcsin(m)
            arccos(m)
            arccosh(m)
            divide(m, 2)
            allclose(m, 0.5)