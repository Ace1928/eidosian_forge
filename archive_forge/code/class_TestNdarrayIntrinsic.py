import random
import numpy as np
from numba.tests.support import TestCase, captured_stdout
from numba import njit, literally
from numba.core import types
from numba.cpython.unsafe.tuple import tuple_setitem, build_full_slice_tuple
from numba.np.unsafe.ndarray import to_fixed_tuple, empty_inferred
from numba.core.unsafe.bytes import memcpy_region
from numba.core.unsafe.refcount import dump_refcount
from numba.cpython.unsafe.numbers import trailing_zeros, leading_zeros
from numba.core.errors import TypingError
class TestNdarrayIntrinsic(TestCase):
    """Tests for numba.unsafe.ndarray
    """

    def test_to_fixed_tuple(self):
        const = 3

        @njit
        def foo(array):
            a = to_fixed_tuple(array, length=1)
            b = to_fixed_tuple(array, 2)
            c = to_fixed_tuple(array, const)
            d = to_fixed_tuple(array, 0)
            return (a, b, c, d)
        np.random.seed(123)
        for _ in range(10):
            arr = np.random.random(3)
            a, b, c, d = foo(arr)
            self.assertEqual(a, tuple(arr[:1]))
            self.assertEqual(b, tuple(arr[:2]))
            self.assertEqual(c, tuple(arr[:3]))
            self.assertEqual(d, ())
        with self.assertRaises(TypingError) as raises:
            foo(np.random.random((1, 2)))
        self.assertIn('Not supported on array.ndim=2', str(raises.exception))

        @njit
        def tuple_with_length(array, length):
            return to_fixed_tuple(array, length)
        with self.assertRaises(TypingError) as raises:
            tuple_with_length(np.random.random(3), 1)
        expectmsg = '*length* argument must be a constant'
        self.assertIn(expectmsg, str(raises.exception))

    def test_issue_3586_variant1(self):

        @njit
        def func():
            S = empty_inferred((10,))
            a = 1.1
            for i in range(len(S)):
                S[i] = a + 2
            return S
        got = func()
        expect = np.asarray([3.1] * 10)
        np.testing.assert_array_equal(got, expect)

    def test_issue_3586_variant2(self):

        @njit
        def func():
            S = empty_inferred((10,))
            a = 1.1
            for i in range(S.size):
                S[i] = a + 2
            return S
        got = func()
        expect = np.asarray([3.1] * 10)
        np.testing.assert_array_equal(got, expect)