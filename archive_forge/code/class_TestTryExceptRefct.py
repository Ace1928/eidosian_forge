import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
class TestTryExceptRefct(MemoryLeakMixin, TestCase):

    def test_list_direct_raise(self):

        @njit
        def udt(n, raise_at):
            lst = typed.List()
            try:
                for i in range(n):
                    if i == raise_at:
                        raise IndexError
                    lst.append(i)
            except Exception:
                return lst
            else:
                return lst
        out = udt(10, raise_at=5)
        self.assertEqual(list(out), list(range(5)))
        out = udt(10, raise_at=10)
        self.assertEqual(list(out), list(range(10)))

    def test_list_indirect_raise(self):

        @njit
        def appender(lst, n, raise_at):
            for i in range(n):
                if i == raise_at:
                    raise IndexError
                lst.append(i)
            return lst

        @njit
        def udt(n, raise_at):
            lst = typed.List()
            lst.append(48657)
            try:
                appender(lst, n, raise_at)
            except Exception:
                return lst
            else:
                return lst
        out = udt(10, raise_at=5)
        self.assertEqual(list(out), [48657] + list(range(5)))
        out = udt(10, raise_at=10)
        self.assertEqual(list(out), [48657] + list(range(10)))

    def test_incompatible_refinement(self):

        @njit
        def udt():
            try:
                lst = typed.List()
                print('A')
                lst.append(0)
                print('B')
                lst.append('fda')
                print('C')
                return lst
            except Exception:
                print('D')
        with self.assertRaises(TypingError) as raises:
            udt()
        self.assertRegex(str(raises.exception), 'Cannot refine type|cannot safely cast unicode_type to int(32|64)')