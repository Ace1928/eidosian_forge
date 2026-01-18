from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
class TestExtend(MemoryLeakMixin, TestCase):
    """Test list extend. """

    def test_list_extend_empty(self):

        @njit
        def foo(items):
            l = listobject.new_list(int32)
            l.extend(items)
            return len(l)
        self.assertEqual(foo((1,)), 1)
        self.assertEqual(foo((1, 2)), 2)
        self.assertEqual(foo((1, 2, 3)), 3)

    def test_list_extend_typing_error_non_iterable(self):
        self.disable_leak_check()

        @njit
        def foo():
            l = listobject.new_list(int32)
            l.extend(1)
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('extend argument must be iterable', str(raises.exception))