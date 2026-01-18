import sys
import warnings
import numpy as np
from numba import njit, literally
from numba import int32, int64, float32, float64
from numba import typeof
from numba.typed import Dict, dictobject, List
from numba.typed.typedobjectutils import _sentry_safe_cast
from numba.core.errors import TypingError
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin, unittest,
from numba.experimental import jitclass
from numba.extending import overload
class TestDictTypeCasting(TestCase):

    def check_good(self, fromty, toty):
        _sentry_safe_cast(fromty, toty)

    def check_bad(self, fromty, toty):
        with self.assertRaises(TypingError) as raises:
            _sentry_safe_cast(fromty, toty)
        self.assertIn('cannot safely cast {fromty} to {toty}'.format(**locals()), str(raises.exception))

    def test_cast_int_to(self):
        self.check_good(types.int32, types.float32)
        self.check_good(types.int32, types.float64)
        self.check_good(types.int32, types.complex128)
        self.check_good(types.int64, types.complex128)
        self.check_bad(types.int32, types.complex64)
        self.check_good(types.int8, types.complex64)

    def test_cast_float_to(self):
        self.check_good(types.float32, types.float64)
        self.check_good(types.float32, types.complex64)
        self.check_good(types.float64, types.complex128)

    def test_cast_bool_to(self):
        self.check_good(types.boolean, types.int32)
        self.check_good(types.boolean, types.float64)
        self.check_good(types.boolean, types.complex128)