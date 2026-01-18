from itertools import product
from itertools import permutations
from numba import njit, typeof
from numba.core import types
import unittest
from numba.tests.support import (TestCase, no_pyobj_flags, MemoryLeakMixin)
from numba.core.errors import TypingError, UnsupportedError
from numba.cpython.unicode import _MAX_UNICODE
from numba.core.types.functions import _header_lead
from numba.extending import overload
class TestUnicodeIteration(BaseTest):

    def test_unicode_iter(self):
        pyfunc = iter_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            self.assertPreciseEqual(pyfunc(a), cfunc(a))

    def test_unicode_literal_iter(self):
        pyfunc = literal_iter_usecase
        cfunc = njit(pyfunc)
        self.assertPreciseEqual(pyfunc(), cfunc())

    def test_unicode_enumerate_iter(self):
        pyfunc = enumerated_iter_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            self.assertPreciseEqual(pyfunc(a), cfunc(a))

    def test_unicode_stopiteration_iter(self):
        self.disable_leak_check()
        pyfunc = iter_stopiteration_usecase
        cfunc = njit(pyfunc)
        for f in (pyfunc, cfunc):
            for a in UNICODE_EXAMPLES:
                with self.assertRaises(StopIteration):
                    f(a)

    def test_unicode_literal_stopiteration_iter(self):
        pyfunc = literal_iter_stopiteration_usecase
        cfunc = njit(pyfunc)
        for f in (pyfunc, cfunc):
            with self.assertRaises(StopIteration):
                f()