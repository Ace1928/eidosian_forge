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
class TestTupleIntrinsic(TestCase):
    """Tests for numba.unsafe.tuple
    """

    def test_tuple_setitem(self):

        @njit
        def foo(tup, idxs, vals):
            out_tup = tup
            for i, v in zip(idxs, vals):
                out_tup = tuple_setitem(out_tup, i, v)
            return (tup, out_tup)
        random.seed(123)
        for _ in range(20):
            n = random.randint(1, 10)
            tup = tuple([random.randint(0, n) for i in range(n)])
            vals = tuple([random.randint(10, 20) for i in range(n)])
            idxs = list(range(len(vals)))
            random.shuffle(idxs)
            idxs = tuple(idxs)
            expect_tup = tuple(tup)
            expect_out = np.asarray(expect_tup)
            expect_out[np.asarray(idxs)] = vals
            got_tup, got_out = foo(tup, idxs, vals)
            self.assertEqual(got_tup, expect_tup)
            self.assertEqual(got_out, tuple(expect_out))

    def test_slice_tuple(self):

        @njit
        def full_slice_array(a, n):
            return a[build_full_slice_tuple(literally(n))]
        for n in range(1, 3):
            a = np.random.random(np.arange(n) + 1)
            for i in range(1, n + 1):
                np.testing.assert_array_equal(a, full_slice_array(a, i))
            with self.assertRaises(TypingError):
                full_slice_array(a, n + 1)