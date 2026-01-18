import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
class TestNonzero:

    def test_nonzero_trivial(self):
        assert_equal(np.count_nonzero(np.array([])), 0)
        assert_equal(np.count_nonzero(np.array([], dtype='?')), 0)
        assert_equal(np.nonzero(np.array([])), ([],))
        assert_equal(np.count_nonzero(np.array([0])), 0)
        assert_equal(np.count_nonzero(np.array([0], dtype='?')), 0)
        assert_equal(np.nonzero(np.array([0])), ([],))
        assert_equal(np.count_nonzero(np.array([1])), 1)
        assert_equal(np.count_nonzero(np.array([1], dtype='?')), 1)
        assert_equal(np.nonzero(np.array([1])), ([0],))

    def test_nonzero_zerod(self):
        assert_equal(np.count_nonzero(np.array(0)), 0)
        assert_equal(np.count_nonzero(np.array(0, dtype='?')), 0)
        with assert_warns(DeprecationWarning):
            assert_equal(np.nonzero(np.array(0)), ([],))
        assert_equal(np.count_nonzero(np.array(1)), 1)
        assert_equal(np.count_nonzero(np.array(1, dtype='?')), 1)
        with assert_warns(DeprecationWarning):
            assert_equal(np.nonzero(np.array(1)), ([0],))

    def test_nonzero_onedim(self):
        x = np.array([1, 0, 2, -1, 0, 0, 8])
        assert_equal(np.count_nonzero(x), 4)
        assert_equal(np.count_nonzero(x), 4)
        assert_equal(np.nonzero(x), ([0, 2, 3, 6],))
        x = np.array([(1, 2, -5, -3), (0, 0, 2, 7), (1, 1, 0, 1), (-1, 3, 1, 0), (0, 7, 0, 4)], dtype=[('a', 'i4'), ('b', 'i2'), ('c', 'i1'), ('d', 'i8')])
        assert_equal(np.count_nonzero(x['a']), 3)
        assert_equal(np.count_nonzero(x['b']), 4)
        assert_equal(np.count_nonzero(x['c']), 3)
        assert_equal(np.count_nonzero(x['d']), 4)
        assert_equal(np.nonzero(x['a']), ([0, 2, 3],))
        assert_equal(np.nonzero(x['b']), ([0, 2, 3, 4],))

    def test_nonzero_twodim(self):
        x = np.array([[0, 1, 0], [2, 0, 3]])
        assert_equal(np.count_nonzero(x.astype('i1')), 3)
        assert_equal(np.count_nonzero(x.astype('i2')), 3)
        assert_equal(np.count_nonzero(x.astype('i4')), 3)
        assert_equal(np.count_nonzero(x.astype('i8')), 3)
        assert_equal(np.nonzero(x), ([0, 1, 1], [1, 0, 2]))
        x = np.eye(3)
        assert_equal(np.count_nonzero(x.astype('i1')), 3)
        assert_equal(np.count_nonzero(x.astype('i2')), 3)
        assert_equal(np.count_nonzero(x.astype('i4')), 3)
        assert_equal(np.count_nonzero(x.astype('i8')), 3)
        assert_equal(np.nonzero(x), ([0, 1, 2], [0, 1, 2]))
        x = np.array([[(0, 1), (0, 0), (1, 11)], [(1, 1), (1, 0), (0, 0)], [(0, 0), (1, 5), (0, 1)]], dtype=[('a', 'f4'), ('b', 'u1')])
        assert_equal(np.count_nonzero(x['a']), 4)
        assert_equal(np.count_nonzero(x['b']), 5)
        assert_equal(np.nonzero(x['a']), ([0, 1, 1, 2], [2, 0, 1, 1]))
        assert_equal(np.nonzero(x['b']), ([0, 0, 1, 2, 2], [0, 2, 0, 1, 2]))
        assert_(not x['a'].T.flags.aligned)
        assert_equal(np.count_nonzero(x['a'].T), 4)
        assert_equal(np.count_nonzero(x['b'].T), 5)
        assert_equal(np.nonzero(x['a'].T), ([0, 1, 1, 2], [1, 1, 2, 0]))
        assert_equal(np.nonzero(x['b'].T), ([0, 0, 1, 2, 2], [0, 1, 2, 0, 2]))

    def test_sparse(self):
        for i in range(20):
            c = np.zeros(200, dtype=bool)
            c[i::20] = True
            assert_equal(np.nonzero(c)[0], np.arange(i, 200 + i, 20))
            c = np.zeros(400, dtype=bool)
            c[10 + i:20 + i] = True
            c[20 + i * 2] = True
            assert_equal(np.nonzero(c)[0], np.concatenate((np.arange(10 + i, 20 + i), [20 + i * 2])))

    def test_return_type(self):

        class C(np.ndarray):
            pass
        for view in (C, np.ndarray):
            for nd in range(1, 4):
                shape = tuple(range(2, 2 + nd))
                x = np.arange(np.prod(shape)).reshape(shape).view(view)
                for nzx in (np.nonzero(x), x.nonzero()):
                    for nzx_i in nzx:
                        assert_(type(nzx_i) is np.ndarray)
                        assert_(nzx_i.flags.writeable)

    def test_count_nonzero_axis(self):
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])
        expected = np.array([1, 1, 1, 1, 1])
        assert_equal(np.count_nonzero(m, axis=0), expected)
        expected = np.array([2, 3])
        assert_equal(np.count_nonzero(m, axis=1), expected)
        assert_raises(ValueError, np.count_nonzero, m, axis=(1, 1))
        assert_raises(TypeError, np.count_nonzero, m, axis='foo')
        assert_raises(np.AxisError, np.count_nonzero, m, axis=3)
        assert_raises(TypeError, np.count_nonzero, m, axis=np.array([[1], [2]]))

    def test_count_nonzero_axis_all_dtypes(self):
        msg = 'Mismatch for dtype: %s'

        def assert_equal_w_dt(a, b, err_msg):
            assert_equal(a.dtype, b.dtype, err_msg=err_msg)
            assert_equal(a, b, err_msg=err_msg)
        for dt in np.typecodes['All']:
            err_msg = msg % (np.dtype(dt).name,)
            if dt != 'V':
                if dt != 'M':
                    m = np.zeros((3, 3), dtype=dt)
                    n = np.ones(1, dtype=dt)
                    m[0, 0] = n[0]
                    m[1, 0] = n[0]
                else:
                    m = np.array(['1970-01-01'] * 9)
                    m = m.reshape((3, 3))
                    m[0, 0] = '1970-01-12'
                    m[1, 0] = '1970-01-12'
                    m = m.astype(dt)
                expected = np.array([2, 0, 0], dtype=np.intp)
                assert_equal_w_dt(np.count_nonzero(m, axis=0), expected, err_msg=err_msg)
                expected = np.array([1, 1, 0], dtype=np.intp)
                assert_equal_w_dt(np.count_nonzero(m, axis=1), expected, err_msg=err_msg)
                expected = np.array(2)
                assert_equal(np.count_nonzero(m, axis=(0, 1)), expected, err_msg=err_msg)
                assert_equal(np.count_nonzero(m, axis=None), expected, err_msg=err_msg)
                assert_equal(np.count_nonzero(m), expected, err_msg=err_msg)
            if dt == 'V':
                m = np.array([np.void(1)] * 6).reshape((2, 3))
                expected = np.array([0, 0, 0], dtype=np.intp)
                assert_equal_w_dt(np.count_nonzero(m, axis=0), expected, err_msg=err_msg)
                expected = np.array([0, 0], dtype=np.intp)
                assert_equal_w_dt(np.count_nonzero(m, axis=1), expected, err_msg=err_msg)
                expected = np.array(0)
                assert_equal(np.count_nonzero(m, axis=(0, 1)), expected, err_msg=err_msg)
                assert_equal(np.count_nonzero(m, axis=None), expected, err_msg=err_msg)
                assert_equal(np.count_nonzero(m), expected, err_msg=err_msg)

    def test_count_nonzero_axis_consistent(self):
        from itertools import combinations, permutations
        axis = (0, 1, 2, 3)
        size = (5, 5, 5, 5)
        msg = 'Mismatch for axis: %s'
        rng = np.random.RandomState(1234)
        m = rng.randint(-100, 100, size=size)
        n = m.astype(object)
        for length in range(len(axis)):
            for combo in combinations(axis, length):
                for perm in permutations(combo):
                    assert_equal(np.count_nonzero(m, axis=perm), np.count_nonzero(n, axis=perm), err_msg=msg % (perm,))

    def test_countnonzero_axis_empty(self):
        a = np.array([[0, 0, 1], [1, 0, 1]])
        assert_equal(np.count_nonzero(a, axis=()), a.astype(bool))

    def test_countnonzero_keepdims(self):
        a = np.array([[0, 0, 1, 0], [0, 3, 5, 0], [7, 9, 2, 0]])
        assert_equal(np.count_nonzero(a, axis=0, keepdims=True), [[1, 2, 3, 0]])
        assert_equal(np.count_nonzero(a, axis=1, keepdims=True), [[1], [2], [3]])
        assert_equal(np.count_nonzero(a, keepdims=True), [[6]])

    def test_array_method(self):
        m = np.array([[1, 0, 0], [4, 0, 6]])
        tgt = [[0, 1, 1], [0, 0, 2]]
        assert_equal(m.nonzero(), tgt)

    def test_nonzero_invalid_object(self):
        a = np.array([np.array([1, 2]), 3], dtype=object)
        assert_raises(ValueError, np.nonzero, a)

        class BoolErrors:

            def __bool__(self):
                raise ValueError('Not allowed')
        assert_raises(ValueError, np.nonzero, np.array([BoolErrors()]))

    def test_nonzero_sideeffect_safety(self):

        class FalseThenTrue:
            _val = False

            def __bool__(self):
                try:
                    return self._val
                finally:
                    self._val = True

        class TrueThenFalse:
            _val = True

            def __bool__(self):
                try:
                    return self._val
                finally:
                    self._val = False
        a = np.array([True, FalseThenTrue()])
        assert_raises(RuntimeError, np.nonzero, a)
        a = np.array([[True], [FalseThenTrue()]])
        assert_raises(RuntimeError, np.nonzero, a)
        a = np.array([False, TrueThenFalse()])
        assert_raises(RuntimeError, np.nonzero, a)
        a = np.array([[False], [TrueThenFalse()]])
        assert_raises(RuntimeError, np.nonzero, a)

    def test_nonzero_sideffects_structured_void(self):
        arr = np.zeros(5, dtype='i1,i8,i8')
        assert arr.flags.aligned
        assert not arr['f2'].flags.aligned
        np.nonzero(arr)
        assert arr.flags.aligned
        np.count_nonzero(arr)
        assert arr.flags.aligned

    def test_nonzero_exception_safe(self):

        class ThrowsAfter:

            def __init__(self, iters):
                self.iters_left = iters

            def __bool__(self):
                if self.iters_left == 0:
                    raise ValueError('called `iters` times')
                self.iters_left -= 1
                return True
        '\n        Test that a ValueError is raised instead of a SystemError\n\n        If the __bool__ function is called after the error state is set,\n        Python (cpython) will raise a SystemError.\n        '
        a = np.array([ThrowsAfter(5)] * 10)
        assert_raises(ValueError, np.nonzero, a)
        a = np.array([ThrowsAfter(15)] * 10)
        assert_raises(ValueError, np.nonzero, a)
        a = np.array([[ThrowsAfter(15)]] * 10)
        assert_raises(ValueError, np.nonzero, a)

    @pytest.mark.skipif(IS_WASM, reason="wasm doesn't have threads")
    def test_structured_threadsafety(self):
        from concurrent.futures import ThreadPoolExecutor
        dt = np.dtype([('', 'f8')])
        dt = np.dtype([('', dt)])
        dt = np.dtype([('', dt)] * 2)
        arr = np.random.uniform(size=(5000, 4)).view(dt)[:, 0]

        def func(arr):
            arr.nonzero()
        tpe = ThreadPoolExecutor(max_workers=8)
        futures = [tpe.submit(func, arr) for _ in range(10)]
        for f in futures:
            f.result()
        assert arr.dtype is dt