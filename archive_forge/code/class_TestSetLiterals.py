import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
class TestSetLiterals(BaseTest):

    def check(self, pyfunc):
        cfunc = njit(pyfunc)
        expected = pyfunc()
        got = cfunc()
        self.assertPreciseEqual(expected, got)
        return (got, expected)

    def test_build_set(self):
        pyfunc = set_literal_return_usecase((1, 2, 3, 2))
        self.check(pyfunc)

    def test_build_heterogeneous_set(self, flags=enable_pyobj_flags):
        pyfunc = set_literal_return_usecase((1, 2.0, 3j, 2))
        self.check(pyfunc)
        pyfunc = set_literal_return_usecase((2.0, 2))
        got, expected = self.check(pyfunc)
        self.assertIs(type(got.pop()), type(expected.pop()))

    def test_build_set_nopython(self):
        arg = list(self.sparse_array(50))
        pyfunc = set_literal_convert_usecase(arg)
        cfunc = jit(nopython=True)(pyfunc)
        expected = pyfunc()
        got = cfunc()
        self.assertPreciseEqual(sorted(expected), sorted(got))