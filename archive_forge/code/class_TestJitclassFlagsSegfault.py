import numpy as np
import unittest
from numba.np.numpy_support import from_dtype
from numba import njit, typeof
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba.core.errors import TypingError
from numba.experimental import jitclass
class TestJitclassFlagsSegfault(MemoryLeakMixin, TestCase):
    """Regression test for: https://github.com/numba/numba/issues/4775 """

    def test(self):

        @jitclass(dict())
        class B(object):

            def __init__(self):
                pass

            def foo(self, X):
                X.flags
        Z = B()
        Z.foo(np.ones(4))