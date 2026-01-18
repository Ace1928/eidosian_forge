import numpy as np
import unittest
from numba import jit, vectorize, int8, int16, int32
from numba.tests.support import TestCase
from numba.tests.enum_usecases import (Color, Shape, Shake,
class BaseEnumTest(object):

    def test_compare(self):
        pyfunc = compare_usecase
        cfunc = jit(nopython=True)(pyfunc)
        for args in self.pairs:
            self.assertPreciseEqual(pyfunc(*args), cfunc(*args))

    def test_return(self):
        """
        Passing and returning enum members.
        """
        pyfunc = return_usecase
        cfunc = jit(nopython=True)(pyfunc)
        for pair in self.pairs:
            for pred in (True, False):
                args = pair + (pred,)
                self.assertIs(pyfunc(*args), cfunc(*args))

    def check_constant_usecase(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)
        for arg in self.values:
            self.assertPreciseEqual(pyfunc(arg), cfunc(arg))

    def test_constant(self):
        self.check_constant_usecase(getattr_usecase)
        self.check_constant_usecase(getitem_usecase)
        self.check_constant_usecase(make_constant_usecase(self.values[0]))