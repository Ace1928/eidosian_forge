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
class TestUnicodeAuxillary(BaseTest):

    def test_ord(self):
        pyfunc = ord_usecase
        cfunc = njit(pyfunc)
        for ex in UNICODE_EXAMPLES:
            for a in ex:
                self.assertPreciseEqual(pyfunc(a), cfunc(a))

    def test_ord_invalid(self):
        self.disable_leak_check()
        pyfunc = ord_usecase
        cfunc = njit(pyfunc)
        for func in (pyfunc, cfunc):
            for ch in ('', 'abc'):
                with self.assertRaises(TypeError) as raises:
                    func(ch)
                self.assertIn('ord() expected a character', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc(1.23)
        self.assertIn(_header_lead, str(raises.exception))

    def test_chr(self):
        pyfunc = chr_usecase
        cfunc = njit(pyfunc)
        for ex in UNICODE_EXAMPLES:
            for x in ex:
                a = ord(x)
                self.assertPreciseEqual(pyfunc(a), cfunc(a))
        for a in (0, _MAX_UNICODE):
            self.assertPreciseEqual(pyfunc(a), cfunc(a))

    def test_chr_invalid(self):
        pyfunc = chr_usecase
        cfunc = njit(pyfunc)
        for func in (pyfunc, cfunc):
            for v in (-2, _MAX_UNICODE + 1):
                with self.assertRaises(ValueError) as raises:
                    func(v)
                self.assertIn('chr() arg not in range', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc('abc')
        self.assertIn(_header_lead, str(raises.exception))

    def test_unicode_type_mro(self):

        def bar(x):
            return True

        @overload(bar)
        def ol_bar(x):
            ok = False
            if isinstance(x, types.UnicodeType):
                if isinstance(x, types.Hashable):
                    ok = True
            return lambda x: ok

        @njit
        def foo(strinst):
            return bar(strinst)
        inst = 'abc'
        self.assertEqual(foo.py_func(inst), foo(inst))
        self.assertIn(types.Hashable, types.unicode_type.__class__.__mro__)

    def test_f_strings(self):
        """test f-string support, which requires bytecode handling
        """

        def impl1(a):
            return f'AA_{a + 3}_B'

        def impl2(a):
            return f'{a + 2}'

        def impl3(a):
            return f'ABC_{a}'

        def impl4(a):
            return f'ABC_{a:0}'

        def impl5():
            return f''
        self.assertEqual(impl1(3), njit(impl1)(3))
        self.assertEqual(impl2(2), njit(impl2)(2))
        self.assertEqual(impl3('DE'), njit(impl3)('DE'))
        list_arg = ['A', 'B']
        got = njit(impl3)(list_arg)
        expected = f'ABC_<object type:{typeof(list_arg)}>'
        self.assertEqual(got, expected)
        with self.assertRaises(UnsupportedError) as raises:
            njit(impl4)(['A', 'B'])
        msg = 'format spec in f-strings not supported yet'
        self.assertIn(msg, str(raises.exception))
        self.assertEqual(impl5(), njit(impl5)())