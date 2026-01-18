from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
class TestReverse(MemoryLeakMixin, TestCase):
    """Test list reverse. """

    def test_list_reverse_empty(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            l.reverse()
            return len(l)
        self.assertEqual(foo(), 0)

    def test_list_reverse_singleton(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            l.append(0)
            l.reverse()
            return (len(l), l[0])
        self.assertEqual(foo(), (1, 0))

    def test_list_reverse_multiple(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10, 13):
                l.append(j)
            l.reverse()
            return (len(l), l[0], l[1], l[2])
        self.assertEqual(foo(), (3, 12, 11, 10))