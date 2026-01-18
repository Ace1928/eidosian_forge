import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
class TestMaskedArrayMethods:

    def setup_method(self):
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928, 8.43, 7.78, 9.865, 5.878, 8.979, 4.732, 3.012, 6.022, 5.095, 3.116, 5.238, 3.957, 6.04, 9.63, 7.712, 3.382, 4.489, 6.479, 7.189, 9.645, 5.395, 4.961, 9.894, 2.893, 7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        X = x.reshape(6, 6)
        XX = x.reshape(3, 2, 2, 3)
        m = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0])
        mx = array(data=x, mask=m)
        mX = array(data=X, mask=m.reshape(X.shape))
        mXX = array(data=XX, mask=m.reshape(XX.shape))
        m2 = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1])
        m2x = array(data=x, mask=m2)
        m2X = array(data=X, mask=m2.reshape(X.shape))
        m2XX = array(data=XX, mask=m2.reshape(XX.shape))
        self.d = (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX)

    def test_generic_methods(self):
        a = array([1, 3, 2])
        assert_equal(a.any(), a._data.any())
        assert_equal(a.all(), a._data.all())
        assert_equal(a.argmax(), a._data.argmax())
        assert_equal(a.argmin(), a._data.argmin())
        assert_equal(a.choose(0, 1, 2, 3, 4), a._data.choose(0, 1, 2, 3, 4))
        assert_equal(a.compress([1, 0, 1]), a._data.compress([1, 0, 1]))
        assert_equal(a.conj(), a._data.conj())
        assert_equal(a.conjugate(), a._data.conjugate())
        m = array([[1, 2], [3, 4]])
        assert_equal(m.diagonal(), m._data.diagonal())
        assert_equal(a.sum(), a._data.sum())
        assert_equal(a.take([1, 2]), a._data.take([1, 2]))
        assert_equal(m.transpose(), m._data.transpose())

    def test_allclose(self):
        a = np.random.rand(10)
        b = a + np.random.rand(10) * 1e-08
        assert_(allclose(a, b))
        a[0] = np.inf
        assert_(not allclose(a, b))
        b[0] = np.inf
        assert_(allclose(a, b))
        a = masked_array(a)
        a[-1] = masked
        assert_(allclose(a, b, masked_equal=True))
        assert_(not allclose(a, b, masked_equal=False))
        a *= 1e-08
        a[0] = 0
        assert_(allclose(a, 0, masked_equal=True))
        a = masked_array([np.iinfo(np.int_).min], dtype=np.int_)
        assert_(allclose(a, a))

    def test_allclose_timedelta(self):
        a = np.array([[1, 2, 3, 4]], dtype='m8[ns]')
        assert allclose(a, a, atol=0)
        assert allclose(a, a, atol=np.timedelta64(1, 'ns'))

    def test_allany(self):
        x = np.array([[0.13, 0.26, 0.9], [0.28, 0.33, 0.63], [0.31, 0.87, 0.7]])
        m = np.array([[True, False, False], [False, False, False], [True, True, False]], dtype=np.bool_)
        mx = masked_array(x, mask=m)
        mxbig = mx > 0.5
        mxsmall = mx < 0.5
        assert_(not mxbig.all())
        assert_(mxbig.any())
        assert_equal(mxbig.all(0), [False, False, True])
        assert_equal(mxbig.all(1), [False, False, True])
        assert_equal(mxbig.any(0), [False, False, True])
        assert_equal(mxbig.any(1), [True, True, True])
        assert_(not mxsmall.all())
        assert_(mxsmall.any())
        assert_equal(mxsmall.all(0), [True, True, False])
        assert_equal(mxsmall.all(1), [False, False, False])
        assert_equal(mxsmall.any(0), [True, True, False])
        assert_equal(mxsmall.any(1), [True, True, False])

    def test_allany_oddities(self):
        store = empty((), dtype=bool)
        full = array([1, 2, 3], mask=True)
        assert_(full.all() is masked)
        full.all(out=store)
        assert_(store)
        assert_(store._mask, True)
        assert_(store is not masked)
        store = empty((), dtype=bool)
        assert_(full.any() is masked)
        full.any(out=store)
        assert_(not store)
        assert_(store._mask, True)
        assert_(store is not masked)

    def test_argmax_argmin(self):
        x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX = self.d
        assert_equal(mx.argmin(), 35)
        assert_equal(mX.argmin(), 35)
        assert_equal(m2x.argmin(), 4)
        assert_equal(m2X.argmin(), 4)
        assert_equal(mx.argmax(), 28)
        assert_equal(mX.argmax(), 28)
        assert_equal(m2x.argmax(), 31)
        assert_equal(m2X.argmax(), 31)
        assert_equal(mX.argmin(0), [2, 2, 2, 5, 0, 5])
        assert_equal(m2X.argmin(0), [2, 2, 4, 5, 0, 4])
        assert_equal(mX.argmax(0), [0, 5, 0, 5, 4, 0])
        assert_equal(m2X.argmax(0), [5, 5, 0, 5, 1, 0])
        assert_equal(mX.argmin(1), [4, 1, 0, 0, 5, 5])
        assert_equal(m2X.argmin(1), [4, 4, 0, 0, 5, 3])
        assert_equal(mX.argmax(1), [2, 4, 1, 1, 4, 1])
        assert_equal(m2X.argmax(1), [2, 4, 1, 1, 1, 1])

    def test_clip(self):
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928, 8.43, 7.78, 9.865, 5.878, 8.979, 4.732, 3.012, 6.022, 5.095, 3.116, 5.238, 3.957, 6.04, 9.63, 7.712, 3.382, 4.489, 6.479, 7.189, 9.645, 5.395, 4.961, 9.894, 2.893, 7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        m = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0])
        mx = array(x, mask=m)
        clipped = mx.clip(2, 8)
        assert_equal(clipped.mask, mx.mask)
        assert_equal(clipped._data, x.clip(2, 8))
        assert_equal(clipped._data, mx._data.clip(2, 8))

    def test_clip_out(self):
        a = np.arange(10)
        m = np.ma.MaskedArray(a, mask=[0, 1] * 5)
        m.clip(0, 5, out=m)
        assert_equal(m.mask, [0, 1] * 5)

    def test_compress(self):
        a = masked_array([1.0, 2.0, 3.0, 4.0, 5.0], fill_value=9999)
        condition = (a > 1.5) & (a < 3.5)
        assert_equal(a.compress(condition), [2.0, 3.0])
        a[[2, 3]] = masked
        b = a.compress(condition)
        assert_equal(b._data, [2.0, 3.0])
        assert_equal(b._mask, [0, 1])
        assert_equal(b.fill_value, 9999)
        assert_equal(b, a[condition])
        condition = a < 4.0
        b = a.compress(condition)
        assert_equal(b._data, [1.0, 2.0, 3.0])
        assert_equal(b._mask, [0, 0, 1])
        assert_equal(b.fill_value, 9999)
        assert_equal(b, a[condition])
        a = masked_array([[10, 20, 30], [40, 50, 60]], mask=[[0, 0, 1], [1, 0, 0]])
        b = a.compress(a.ravel() >= 22)
        assert_equal(b._data, [30, 40, 50, 60])
        assert_equal(b._mask, [1, 1, 0, 0])
        x = np.array([3, 1, 2])
        b = a.compress(x >= 2, axis=1)
        assert_equal(b._data, [[10, 30], [40, 60]])
        assert_equal(b._mask, [[0, 1], [1, 0]])

    def test_compressed(self):
        a = array([1, 2, 3, 4], mask=[0, 0, 0, 0])
        b = a.compressed()
        assert_equal(b, a)
        a[0] = masked
        b = a.compressed()
        assert_equal(b, [2, 3, 4])

    def test_empty(self):
        datatype = [('a', int), ('b', float), ('c', '|S8')]
        a = masked_array([(1, 1.1, '1.1'), (2, 2.2, '2.2'), (3, 3.3, '3.3')], dtype=datatype)
        assert_equal(len(a.fill_value.item()), len(datatype))
        b = empty_like(a)
        assert_equal(b.shape, a.shape)
        assert_equal(b.fill_value, a.fill_value)
        b = empty(len(a), dtype=datatype)
        assert_equal(b.shape, a.shape)
        assert_equal(b.fill_value, a.fill_value)
        a = masked_array([1, 2, 3], mask=[False, True, False])
        b = empty_like(a)
        assert_(not np.may_share_memory(a.mask, b.mask))
        b = a.view(masked_array)
        assert_(np.may_share_memory(a.mask, b.mask))

    def test_zeros(self):
        datatype = [('a', int), ('b', float), ('c', '|S8')]
        a = masked_array([(1, 1.1, '1.1'), (2, 2.2, '2.2'), (3, 3.3, '3.3')], dtype=datatype)
        assert_equal(len(a.fill_value.item()), len(datatype))
        b = zeros(len(a), dtype=datatype)
        assert_equal(b.shape, a.shape)
        assert_equal(b.fill_value, a.fill_value)
        b = zeros_like(a)
        assert_equal(b.shape, a.shape)
        assert_equal(b.fill_value, a.fill_value)
        a = masked_array([1, 2, 3], mask=[False, True, False])
        b = zeros_like(a)
        assert_(not np.may_share_memory(a.mask, b.mask))
        b = a.view()
        assert_(np.may_share_memory(a.mask, b.mask))

    def test_ones(self):
        datatype = [('a', int), ('b', float), ('c', '|S8')]
        a = masked_array([(1, 1.1, '1.1'), (2, 2.2, '2.2'), (3, 3.3, '3.3')], dtype=datatype)
        assert_equal(len(a.fill_value.item()), len(datatype))
        b = ones(len(a), dtype=datatype)
        assert_equal(b.shape, a.shape)
        assert_equal(b.fill_value, a.fill_value)
        b = ones_like(a)
        assert_equal(b.shape, a.shape)
        assert_equal(b.fill_value, a.fill_value)
        a = masked_array([1, 2, 3], mask=[False, True, False])
        b = ones_like(a)
        assert_(not np.may_share_memory(a.mask, b.mask))
        b = a.view()
        assert_(np.may_share_memory(a.mask, b.mask))

    @suppress_copy_mask_on_assignment
    def test_put(self):
        d = arange(5)
        n = [0, 0, 0, 1, 1]
        m = make_mask(n)
        x = array(d, mask=m)
        assert_(x[3] is masked)
        assert_(x[4] is masked)
        x[[1, 4]] = [10, 40]
        assert_(x[3] is masked)
        assert_(x[4] is not masked)
        assert_equal(x, [0, 10, 2, -1, 40])
        x = masked_array(arange(10), mask=[1, 0, 0, 0, 0] * 2)
        i = [0, 2, 4, 6]
        x.put(i, [6, 4, 2, 0])
        assert_equal(x, asarray([6, 1, 4, 3, 2, 5, 0, 7, 8, 9]))
        assert_equal(x.mask, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        x.put(i, masked_array([0, 2, 4, 6], [1, 0, 1, 0]))
        assert_array_equal(x, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert_equal(x.mask, [1, 0, 0, 0, 1, 1, 0, 0, 0, 0])
        x = masked_array(arange(10), mask=[1, 0, 0, 0, 0] * 2)
        put(x, i, [6, 4, 2, 0])
        assert_equal(x, asarray([6, 1, 4, 3, 2, 5, 0, 7, 8, 9]))
        assert_equal(x.mask, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        put(x, i, masked_array([0, 2, 4, 6], [1, 0, 1, 0]))
        assert_array_equal(x, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert_equal(x.mask, [1, 0, 0, 0, 1, 1, 0, 0, 0, 0])

    def test_put_nomask(self):
        x = zeros(10)
        z = array([3.0, -1.0], mask=[False, True])
        x.put([1, 2], z)
        assert_(x[0] is not masked)
        assert_equal(x[0], 0)
        assert_(x[1] is not masked)
        assert_equal(x[1], 3)
        assert_(x[2] is masked)
        assert_(x[3] is not masked)
        assert_equal(x[3], 0)

    def test_put_hardmask(self):
        d = arange(5)
        n = [0, 0, 0, 1, 1]
        m = make_mask(n)
        xh = array(d + 1, mask=m, hard_mask=True, copy=True)
        xh.put([4, 2, 0, 1, 3], [1, 2, 3, 4, 5])
        assert_equal(xh._data, [3, 4, 2, 4, 5])

    def test_putmask(self):
        x = arange(6) + 1
        mx = array(x, mask=[0, 0, 0, 1, 1, 1])
        mask = [0, 0, 1, 0, 0, 1]
        xx = x.copy()
        putmask(xx, mask, 99)
        assert_equal(xx, [1, 2, 99, 4, 5, 99])
        mxx = mx.copy()
        putmask(mxx, mask, 99)
        assert_equal(mxx._data, [1, 2, 99, 4, 5, 99])
        assert_equal(mxx._mask, [0, 0, 0, 1, 1, 0])
        values = array([10, 20, 30, 40, 50, 60], mask=[1, 1, 1, 0, 0, 0])
        xx = x.copy()
        putmask(xx, mask, values)
        assert_equal(xx._data, [1, 2, 30, 4, 5, 60])
        assert_equal(xx._mask, [0, 0, 1, 0, 0, 0])
        mxx = mx.copy()
        putmask(mxx, mask, values)
        assert_equal(mxx._data, [1, 2, 30, 4, 5, 60])
        assert_equal(mxx._mask, [0, 0, 1, 1, 1, 0])
        mxx = mx.copy()
        mxx.harden_mask()
        putmask(mxx, mask, values)
        assert_equal(mxx, [1, 2, 30, 4, 5, 60])

    def test_ravel(self):
        a = array([[1, 2, 3, 4, 5]], mask=[[0, 1, 0, 0, 0]])
        aravel = a.ravel()
        assert_equal(aravel._mask.shape, aravel.shape)
        a = array([0, 0], mask=[1, 1])
        aravel = a.ravel()
        assert_equal(aravel._mask.shape, a.shape)
        a = array([1, 2, 3, 4], mask=[0, 0, 0, 0], shrink=False)
        assert_equal(a.ravel()._mask, [0, 0, 0, 0])
        a.fill_value = -99
        a.shape = (2, 2)
        ar = a.ravel()
        assert_equal(ar._mask, [0, 0, 0, 0])
        assert_equal(ar._data, [1, 2, 3, 4])
        assert_equal(ar.fill_value, -99)
        assert_equal(a.ravel(order='C'), [1, 2, 3, 4])
        assert_equal(a.ravel(order='F'), [1, 3, 2, 4])

    @pytest.mark.parametrize('order', 'AKCF')
    @pytest.mark.parametrize('data_order', 'CF')
    def test_ravel_order(self, order, data_order):
        arr = np.ones((5, 10), order=data_order)
        arr[0, :] = 0
        mask = np.ones((10, 5), dtype=bool, order=data_order).T
        mask[0, :] = False
        x = array(arr, mask=mask)
        assert x._data.flags.fnc != x._mask.flags.fnc
        assert (x.filled(0) == 0).all()
        raveled = x.ravel(order)
        assert (raveled.filled(0) == 0).all()
        assert_array_equal(arr.ravel(order), x.ravel(order)._data)

    def test_reshape(self):
        x = arange(4)
        x[0] = masked
        y = x.reshape(2, 2)
        assert_equal(y.shape, (2, 2))
        assert_equal(y._mask.shape, (2, 2))
        assert_equal(x.shape, (4,))
        assert_equal(x._mask.shape, (4,))

    def test_sort(self):
        x = array([1, 4, 2, 3], mask=[0, 1, 0, 0], dtype=np.uint8)
        sortedx = sort(x)
        assert_equal(sortedx._data, [1, 2, 3, 4])
        assert_equal(sortedx._mask, [0, 0, 0, 1])
        sortedx = sort(x, endwith=False)
        assert_equal(sortedx._data, [4, 1, 2, 3])
        assert_equal(sortedx._mask, [1, 0, 0, 0])
        x.sort()
        assert_equal(x._data, [1, 2, 3, 4])
        assert_equal(x._mask, [0, 0, 0, 1])
        x = array([1, 4, 2, 3], mask=[0, 1, 0, 0], dtype=np.uint8)
        x.sort(endwith=False)
        assert_equal(x._data, [4, 1, 2, 3])
        assert_equal(x._mask, [1, 0, 0, 0])
        x = [1, 4, 2, 3]
        sortedx = sort(x)
        assert_(not isinstance(sorted, MaskedArray))
        x = array([0, 1, -1, -2, 2], mask=nomask, dtype=np.int8)
        sortedx = sort(x, endwith=False)
        assert_equal(sortedx._data, [-2, -1, 0, 1, 2])
        x = array([0, 1, -1, -2, 2], mask=[0, 1, 0, 0, 1], dtype=np.int8)
        sortedx = sort(x, endwith=False)
        assert_equal(sortedx._data, [1, 2, -2, -1, 0])
        assert_equal(sortedx._mask, [1, 1, 0, 0, 0])
        x = array([0, -1], dtype=np.int8)
        sortedx = sort(x, kind='stable')
        assert_equal(sortedx, array([-1, 0], dtype=np.int8))

    def test_stable_sort(self):
        x = array([1, 2, 3, 1, 2, 3], dtype=np.uint8)
        expected = array([0, 3, 1, 4, 2, 5])
        computed = argsort(x, kind='stable')
        assert_equal(computed, expected)

    def test_argsort_matches_sort(self):
        x = array([1, 4, 2, 3], mask=[0, 1, 0, 0], dtype=np.uint8)
        for kwargs in [dict(), dict(endwith=True), dict(endwith=False), dict(fill_value=2), dict(fill_value=2, endwith=True), dict(fill_value=2, endwith=False)]:
            sortedx = sort(x, **kwargs)
            argsortedx = x[argsort(x, **kwargs)]
            assert_equal(sortedx._data, argsortedx._data)
            assert_equal(sortedx._mask, argsortedx._mask)

    def test_sort_2d(self):
        a = masked_array([[8, 4, 1], [2, 0, 9]])
        a.sort(0)
        assert_equal(a, [[2, 0, 1], [8, 4, 9]])
        a = masked_array([[8, 4, 1], [2, 0, 9]])
        a.sort(1)
        assert_equal(a, [[1, 4, 8], [0, 2, 9]])
        a = masked_array([[8, 4, 1], [2, 0, 9]], mask=[[1, 0, 0], [0, 0, 1]])
        a.sort(0)
        assert_equal(a, [[2, 0, 1], [8, 4, 9]])
        assert_equal(a._mask, [[0, 0, 0], [1, 0, 1]])
        a = masked_array([[8, 4, 1], [2, 0, 9]], mask=[[1, 0, 0], [0, 0, 1]])
        a.sort(1)
        assert_equal(a, [[1, 4, 8], [0, 2, 9]])
        assert_equal(a._mask, [[0, 0, 1], [0, 0, 1]])
        a = masked_array([[[7, 8, 9], [4, 5, 6], [1, 2, 3]], [[1, 2, 3], [7, 8, 9], [4, 5, 6]], [[7, 8, 9], [1, 2, 3], [4, 5, 6]], [[4, 5, 6], [1, 2, 3], [7, 8, 9]]])
        a[a % 4 == 0] = masked
        am = a.copy()
        an = a.filled(99)
        am.sort(0)
        an.sort(0)
        assert_equal(am, an)
        am = a.copy()
        an = a.filled(99)
        am.sort(1)
        an.sort(1)
        assert_equal(am, an)
        am = a.copy()
        an = a.filled(99)
        am.sort(2)
        an.sort(2)
        assert_equal(am, an)

    def test_sort_flexible(self):
        a = array(data=[(3, 3), (3, 2), (2, 2), (2, 1), (1, 0), (1, 1), (1, 2)], mask=[(0, 0), (0, 1), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0)], dtype=[('A', int), ('B', int)])
        mask_last = array(data=[(1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (3, 2), (1, 0)], mask=[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (1, 0)], dtype=[('A', int), ('B', int)])
        mask_first = array(data=[(1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (3, 3)], mask=[(1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (0, 0)], dtype=[('A', int), ('B', int)])
        test = sort(a)
        assert_equal(test, mask_last)
        assert_equal(test.mask, mask_last.mask)
        test = sort(a, endwith=False)
        assert_equal(test, mask_first)
        assert_equal(test.mask, mask_first.mask)
        dt = np.dtype([('v', int, 2)])
        a = a.view(dt)
        test = sort(a)
        test = sort(a, endwith=False)

    def test_argsort(self):
        a = array([1, 5, 2, 4, 3], mask=[1, 0, 0, 1, 0])
        assert_equal(np.argsort(a), argsort(a))

    def test_squeeze(self):
        data = masked_array([[1, 2, 3]])
        assert_equal(data.squeeze(), [1, 2, 3])
        data = masked_array([[1, 2, 3]], mask=[[1, 1, 1]])
        assert_equal(data.squeeze(), [1, 2, 3])
        assert_equal(data.squeeze()._mask, [1, 1, 1])
        arr = np.array([[1]])
        arr_sq = arr.squeeze()
        assert_equal(arr_sq, 1)
        arr_sq[...] = 2
        assert_equal(arr[0, 0], 2)
        m_arr = masked_array([[1]], mask=True)
        m_arr_sq = m_arr.squeeze()
        assert_(m_arr_sq is not np.ma.masked)
        assert_equal(m_arr_sq.mask, True)
        m_arr_sq[...] = 2
        assert_equal(m_arr[0, 0], 2)

    def test_swapaxes(self):
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928, 8.43, 7.78, 9.865, 5.878, 8.979, 4.732, 3.012, 6.022, 5.095, 3.116, 5.238, 3.957, 6.04, 9.63, 7.712, 3.382, 4.489, 6.479, 7.189, 9.645, 5.395, 4.961, 9.894, 2.893, 7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        m = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0])
        mX = array(x, mask=m).reshape(6, 6)
        mXX = mX.reshape(3, 2, 2, 3)
        mXswapped = mX.swapaxes(0, 1)
        assert_equal(mXswapped[-1], mX[:, -1])
        mXXswapped = mXX.swapaxes(0, 2)
        assert_equal(mXXswapped.shape, (2, 2, 3, 3))

    def test_take(self):
        x = masked_array([10, 20, 30, 40], [0, 1, 0, 1])
        assert_equal(x.take([0, 0, 3]), masked_array([10, 10, 40], [0, 0, 1]))
        assert_equal(x.take([0, 0, 3]), x[[0, 0, 3]])
        assert_equal(x.take([[0, 1], [0, 1]]), masked_array([[10, 20], [10, 20]], [[0, 1], [0, 1]]))
        assert_(x[1] is np.ma.masked)
        assert_(x.take(1) is np.ma.masked)
        x = array([[10, 20, 30], [40, 50, 60]], mask=[[0, 0, 1], [1, 0, 0]])
        assert_equal(x.take([0, 2], axis=1), array([[10, 30], [40, 60]], mask=[[0, 1], [1, 0]]))
        assert_equal(take(x, [0, 2], axis=1), array([[10, 30], [40, 60]], mask=[[0, 1], [1, 0]]))

    def test_take_masked_indices(self):
        a = np.array((40, 18, 37, 9, 22))
        indices = np.arange(3)[None, :] + np.arange(5)[:, None]
        mindices = array(indices, mask=indices >= len(a))
        test = take(a, mindices, mode='clip')
        ctrl = array([[40, 18, 37], [18, 37, 9], [37, 9, 22], [9, 22, 22], [22, 22, 22]])
        assert_equal(test, ctrl)
        test = take(a, mindices)
        ctrl = array([[40, 18, 37], [18, 37, 9], [37, 9, 22], [9, 22, 40], [22, 40, 40]])
        ctrl[3, 2] = ctrl[4, 1] = ctrl[4, 2] = masked
        assert_equal(test, ctrl)
        assert_equal(test.mask, ctrl.mask)
        a = array((40, 18, 37, 9, 22), mask=(0, 1, 0, 0, 0))
        test = take(a, mindices)
        ctrl[0, 1] = ctrl[1, 0] = masked
        assert_equal(test, ctrl)
        assert_equal(test.mask, ctrl.mask)

    def test_tolist(self):
        x = array(np.arange(12))
        x[[1, -2]] = masked
        xlist = x.tolist()
        assert_(xlist[1] is None)
        assert_(xlist[-2] is None)
        x.shape = (3, 4)
        xlist = x.tolist()
        ctrl = [[0, None, 2, 3], [4, 5, 6, 7], [8, 9, None, 11]]
        assert_equal(xlist[0], [0, None, 2, 3])
        assert_equal(xlist[1], [4, 5, 6, 7])
        assert_equal(xlist[2], [8, 9, None, 11])
        assert_equal(xlist, ctrl)
        x = array(list(zip([1, 2, 3], [1.1, 2.2, 3.3], ['one', 'two', 'thr'])), dtype=[('a', int), ('b', float), ('c', '|S8')])
        x[-1] = masked
        assert_equal(x.tolist(), [(1, 1.1, b'one'), (2, 2.2, b'two'), (None, None, None)])
        a = array([(1, 2), (3, 4)], mask=[(0, 1), (0, 0)], dtype=[('a', int), ('b', int)])
        test = a.tolist()
        assert_equal(test, [[1, None], [3, 4]])
        a = a[0]
        test = a.tolist()
        assert_equal(test, [1, None])

    def test_tolist_specialcase(self):
        a = array([(0, 1), (2, 3)], dtype=[('a', int), ('b', int)])
        for entry in a:
            for item in entry.tolist():
                assert_(not isinstance(item, np.generic))
        a.mask[0] = (0, 1)
        for entry in a:
            for item in entry.tolist():
                assert_(not isinstance(item, np.generic))

    def test_toflex(self):
        data = arange(10)
        record = data.toflex()
        assert_equal(record['_data'], data._data)
        assert_equal(record['_mask'], data._mask)
        data[[0, 1, 2, -1]] = masked
        record = data.toflex()
        assert_equal(record['_data'], data._data)
        assert_equal(record['_mask'], data._mask)
        ndtype = [('i', int), ('s', '|S3'), ('f', float)]
        data = array([(i, s, f) for i, s, f in zip(np.arange(10), 'ABCDEFGHIJKLM', np.random.rand(10))], dtype=ndtype)
        data[[0, 1, 2, -1]] = masked
        record = data.toflex()
        assert_equal(record['_data'], data._data)
        assert_equal(record['_mask'], data._mask)
        ndtype = np.dtype('int, (2,3)float, float')
        data = array([(i, f, ff) for i, f, ff in zip(np.arange(10), np.random.rand(10), np.random.rand(10))], dtype=ndtype)
        data[[0, 1, 2, -1]] = masked
        record = data.toflex()
        assert_equal_records(record['_data'], data._data)
        assert_equal_records(record['_mask'], data._mask)

    def test_fromflex(self):
        a = array([1, 2, 3])
        test = fromflex(a.toflex())
        assert_equal(test, a)
        assert_equal(test.mask, a.mask)
        a = array([1, 2, 3], mask=[0, 0, 1])
        test = fromflex(a.toflex())
        assert_equal(test, a)
        assert_equal(test.mask, a.mask)
        a = array([(1, 1.0), (2, 2.0), (3, 3.0)], mask=[(1, 0), (0, 0), (0, 1)], dtype=[('A', int), ('B', float)])
        test = fromflex(a.toflex())
        assert_equal(test, a)
        assert_equal(test.data, a.data)

    def test_arraymethod(self):
        marray = masked_array([[1, 2, 3, 4, 5]], mask=[0, 0, 1, 0, 0])
        control = masked_array([[1], [2], [3], [4], [5]], mask=[0, 0, 1, 0, 0])
        assert_equal(marray.T, control)
        assert_equal(marray.transpose(), control)
        assert_equal(MaskedArray.cumsum(marray.T, 0), control.cumsum(0))

    def test_arraymethod_0d(self):
        x = np.ma.array(42, mask=True)
        assert_equal(x.T.mask, x.mask)
        assert_equal(x.T.data, x.data)

    def test_transpose_view(self):
        x = np.ma.array([[1, 2, 3], [4, 5, 6]])
        x[0, 1] = np.ma.masked
        xt = x.T
        xt[1, 0] = 10
        xt[0, 1] = np.ma.masked
        assert_equal(x.data, xt.T.data)
        assert_equal(x.mask, xt.T.mask)

    def test_diagonal_view(self):
        x = np.ma.zeros((3, 3))
        x[0, 0] = 10
        x[1, 1] = np.ma.masked
        x[2, 2] = 20
        xd = x.diagonal()
        x[1, 1] = 15
        assert_equal(xd.mask, x.diagonal().mask)
        assert_equal(xd.data, x.diagonal().data)