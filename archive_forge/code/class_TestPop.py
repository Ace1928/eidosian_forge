from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
class TestPop(MemoryLeakMixin, TestCase):
    """Test list pop. """

    def test_list_pop_singleton(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            l.append(0)
            return (l.pop(), len(l))
        self.assertEqual(foo(), (0, 0))

    def test_list_pop_singleton_index(self):

        @njit
        def foo(i):
            l = listobject.new_list(int32)
            l.append(0)
            return (l.pop(i), len(l))
        self.assertEqual(foo(0), (0, 0))
        self.assertEqual(foo(-1), (0, 0))

    def test_list_pop_multiple(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in (10, 11, 12):
                l.append(j)
            return (l.pop(), len(l))
        self.assertEqual(foo(), (12, 2))

    def test_list_pop_multiple_index(self):

        @njit
        def foo(i):
            l = listobject.new_list(int32)
            for j in (10, 11, 12):
                l.append(j)
            return (l.pop(i), len(l))
        for i, n in ((0, 10), (1, 11), (2, 12)):
            self.assertEqual(foo(i), (n, 2))
        for i, n in ((-3, 10), (-2, 11), (-1, 12)):
            self.assertEqual(foo(i), (n, 2))

    def test_list_pop_integer_types_as_index(self):

        @njit
        def foo(i):
            l = listobject.new_list(int32)
            l.append(0)
            return l.pop(i)
        for t in types.signed_domain:
            self.assertEqual(foo(t(0)), 0)

    def test_list_pop_empty_index_error_no_index(self):
        self.disable_leak_check()

        @njit
        def foo():
            l = listobject.new_list(int32)
            l.pop()
        with self.assertRaises(IndexError) as raises:
            foo()
        self.assertIn('pop from empty list', str(raises.exception))

    def test_list_pop_empty_index_error_with_index(self):
        self.disable_leak_check()

        @njit
        def foo(i):
            l = listobject.new_list(int32)
            l.pop(i)
        with self.assertRaises(IndexError) as raises:
            foo(-1)
        self.assertIn('pop from empty list', str(raises.exception))
        with self.assertRaises(IndexError) as raises:
            foo(0)
        self.assertIn('pop from empty list', str(raises.exception))
        with self.assertRaises(IndexError) as raises:
            foo(1)
        self.assertIn('pop from empty list', str(raises.exception))

    def test_list_pop_mutiple_index_error_with_index(self):
        self.disable_leak_check()

        @njit
        def foo(i):
            l = listobject.new_list(int32)
            for j in (10, 11, 12):
                l.append(j)
            l.pop(i)
        with self.assertRaises(IndexError) as raises:
            foo(-4)
        self.assertIn('list index out of range', str(raises.exception))
        with self.assertRaises(IndexError) as raises:
            foo(3)
        self.assertIn('list index out of range', str(raises.exception))

    def test_list_pop_singleton_typing_error_on_index(self):
        self.disable_leak_check()

        @njit
        def foo(i):
            l = listobject.new_list(int32)
            l.append(0)
            return l.pop(i)
        for i in ('xyz', 1.0, 1j):
            with self.assertRaises(TypingError) as raises:
                foo(i)
            self.assertIn('argument for pop must be an integer', str(raises.exception))