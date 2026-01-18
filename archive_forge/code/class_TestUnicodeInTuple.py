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
class TestUnicodeInTuple(BaseTest):

    def test_const_unicode_in_tuple(self):

        @njit
        def f():
            return ('aa',) < ('bb',)
        self.assertEqual(f.py_func(), f())

        @njit
        def f():
            return ('cc',) < ('bb',)
        self.assertEqual(f.py_func(), f())

    def test_const_unicode_in_hetero_tuple(self):

        @njit
        def f():
            return ('aa', 1) < ('bb', 1)
        self.assertEqual(f.py_func(), f())

        @njit
        def f():
            return ('aa', 1) < ('aa', 2)
        self.assertEqual(f.py_func(), f())

    def test_ascii_flag_unbox(self):

        @njit
        def f(s):
            return s._is_ascii
        for s in UNICODE_EXAMPLES:
            self.assertEqual(f(s), isascii(s))

    def test_ascii_flag_join(self):

        @njit
        def f():
            s1 = 'abc'
            s2 = '123'
            s3 = '🐍⚡'
            s4 = '大处着眼，小处着手。'
            return (','.join([s1, s2])._is_ascii, '🐍⚡'.join([s1, s2])._is_ascii, ','.join([s1, s3])._is_ascii, ','.join([s3, s4])._is_ascii)
        self.assertEqual(f(), (1, 0, 0, 0))

    def test_ascii_flag_getitem(self):

        @njit
        def f():
            s1 = 'abc123'
            s2 = '🐍⚡🐍⚡🐍⚡'
            return (s1[0]._is_ascii, s1[2:]._is_ascii, s2[0]._is_ascii, s2[2:]._is_ascii)
        self.assertEqual(f(), (1, 1, 0, 0))

    def test_ascii_flag_add_mul(self):

        @njit
        def f():
            s1 = 'abc'
            s2 = '123'
            s3 = '🐍⚡'
            s4 = '大处着眼，小处着手。'
            return ((s1 + s2)._is_ascii, (s1 + s3)._is_ascii, (s3 + s4)._is_ascii, (s1 * 2)._is_ascii, (s3 * 2)._is_ascii)
        self.assertEqual(f(), (1, 0, 0, 1, 0))