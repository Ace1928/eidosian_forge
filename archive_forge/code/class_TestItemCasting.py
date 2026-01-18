from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
class TestItemCasting(TestCase):

    @njit
    def foo(fromty, toty):
        l = listobject.new_list(toty)
        l.append(fromty(0))

    def check_good(self, fromty, toty):
        TestItemCasting.foo(fromty, toty)

    def check_bad(self, fromty, toty):
        with self.assertRaises(TypingError) as raises:
            TestItemCasting.foo(fromty, toty)
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

    def test_cast_fail_unicode_int(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            l.append('xyz')
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('cannot safely cast unicode_type to int32', str(raises.exception))

    def test_cast_fail_int_unicode(self):

        @njit
        def foo():
            l = listobject.new_list(types.unicode_type)
            l.append(int32(0))
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('Cannot cast int32 to unicode_type', str(raises.exception))