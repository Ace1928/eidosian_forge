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
class TestMaskedArrayMathMethods:

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

    def test_cumsumprod(self):
        x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX = self.d
        mXcp = mX.cumsum(0)
        assert_equal(mXcp._data, mX.filled(0).cumsum(0))
        mXcp = mX.cumsum(1)
        assert_equal(mXcp._data, mX.filled(0).cumsum(1))
        mXcp = mX.cumprod(0)
        assert_equal(mXcp._data, mX.filled(1).cumprod(0))
        mXcp = mX.cumprod(1)
        assert_equal(mXcp._data, mX.filled(1).cumprod(1))

    def test_cumsumprod_with_output(self):
        xm = array(np.random.uniform(0, 10, 12)).reshape(3, 4)
        xm[:, 0] = xm[0] = xm[-1, -1] = masked
        for funcname in ('cumsum', 'cumprod'):
            npfunc = getattr(np, funcname)
            xmmeth = getattr(xm, funcname)
            output = np.empty((3, 4), dtype=float)
            output.fill(-9999)
            result = npfunc(xm, axis=0, out=output)
            assert_(result is output)
            assert_equal(result, xmmeth(axis=0, out=output))
            output = empty((3, 4), dtype=int)
            result = xmmeth(axis=0, out=output)
            assert_(result is output)

    def test_ptp(self):
        x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX = self.d
        n, m = X.shape
        assert_equal(mx.ptp(), mx.compressed().ptp())
        rows = np.zeros(n, float)
        cols = np.zeros(m, float)
        for k in range(m):
            cols[k] = mX[:, k].compressed().ptp()
        for k in range(n):
            rows[k] = mX[k].compressed().ptp()
        assert_equal(mX.ptp(0), cols)
        assert_equal(mX.ptp(1), rows)

    def test_add_object(self):
        x = masked_array(['a', 'b'], mask=[1, 0], dtype=object)
        y = x + 'x'
        assert_equal(y[1], 'bx')
        assert_(y.mask[0])

    def test_sum_object(self):
        a = masked_array([1, 2, 3], mask=[1, 0, 0], dtype=object)
        assert_equal(a.sum(), 5)
        a = masked_array([[1, 2, 3], [4, 5, 6]], dtype=object)
        assert_equal(a.sum(axis=0), [5, 7, 9])

    def test_prod_object(self):
        a = masked_array([1, 2, 3], mask=[1, 0, 0], dtype=object)
        assert_equal(a.prod(), 2 * 3)
        a = masked_array([[1, 2, 3], [4, 5, 6]], dtype=object)
        assert_equal(a.prod(axis=0), [4, 10, 18])

    def test_meananom_object(self):
        a = masked_array([1, 2, 3], dtype=object)
        assert_equal(a.mean(), 2)
        assert_equal(a.anom(), [-1, 0, 1])

    def test_anom_shape(self):
        a = masked_array([1, 2, 3])
        assert_equal(a.anom().shape, a.shape)
        a.mask = True
        assert_equal(a.anom().shape, a.shape)
        assert_(np.ma.is_masked(a.anom()))

    def test_anom(self):
        a = masked_array(np.arange(1, 7).reshape(2, 3))
        assert_almost_equal(a.anom(), [[-2.5, -1.5, -0.5], [0.5, 1.5, 2.5]])
        assert_almost_equal(a.anom(axis=0), [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        assert_almost_equal(a.anom(axis=1), [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
        a.mask = [[0, 0, 1], [0, 1, 0]]
        mval = -99
        assert_almost_equal(a.anom().filled(mval), [[-2.25, -1.25, mval], [0.75, mval, 2.75]])
        assert_almost_equal(a.anom(axis=0).filled(mval), [[-1.5, 0.0, mval], [1.5, mval, 0.0]])
        assert_almost_equal(a.anom(axis=1).filled(mval), [[-0.5, 0.5, mval], [-1.0, mval, 1.0]])

    def test_trace(self):
        x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX = self.d
        mXdiag = mX.diagonal()
        assert_equal(mX.trace(), mX.diagonal().compressed().sum())
        assert_almost_equal(mX.trace(), X.trace() - sum(mXdiag.mask * X.diagonal(), axis=0))
        assert_equal(np.trace(mX), mX.trace())
        arr = np.arange(2 * 4 * 4).reshape(2, 4, 4)
        m_arr = np.ma.masked_array(arr, False)
        assert_equal(arr.trace(axis1=1, axis2=2), m_arr.trace(axis1=1, axis2=2))

    def test_dot(self):
        x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX = self.d
        fx = mx.filled(0)
        r = mx.dot(mx)
        assert_almost_equal(r.filled(0), fx.dot(fx))
        assert_(r.mask is nomask)
        fX = mX.filled(0)
        r = mX.dot(mX)
        assert_almost_equal(r.filled(0), fX.dot(fX))
        assert_(r.mask[1, 3])
        r1 = empty_like(r)
        mX.dot(mX, out=r1)
        assert_almost_equal(r, r1)
        mYY = mXX.swapaxes(-1, -2)
        fXX, fYY = (mXX.filled(0), mYY.filled(0))
        r = mXX.dot(mYY)
        assert_almost_equal(r.filled(0), fXX.dot(fYY))
        r1 = empty_like(r)
        mXX.dot(mYY, out=r1)
        assert_almost_equal(r, r1)

    def test_dot_shape_mismatch(self):
        x = masked_array([[1, 2], [3, 4]], mask=[[0, 1], [0, 0]])
        y = masked_array([[1, 2], [3, 4]], mask=[[0, 1], [0, 0]])
        z = masked_array([[0, 1], [3, 3]])
        x.dot(y, out=z)
        assert_almost_equal(z.filled(0), [[1, 0], [15, 16]])
        assert_almost_equal(z.mask, [[0, 1], [0, 0]])

    def test_varmean_nomask(self):
        foo = array([1, 2, 3, 4], dtype='f8')
        bar = array([1, 2, 3, 4], dtype='f8')
        assert_equal(type(foo.mean()), np.float64)
        assert_equal(type(foo.var()), np.float64)
        assert (foo.mean() == bar.mean()) is np.bool_(True)
        foo = array(np.arange(16).reshape((4, 4)), dtype='f8')
        bar = empty(4, dtype='f4')
        assert_equal(type(foo.mean(axis=1)), MaskedArray)
        assert_equal(type(foo.var(axis=1)), MaskedArray)
        assert_(foo.mean(axis=1, out=bar) is bar)
        assert_(foo.var(axis=1, out=bar) is bar)

    def test_varstd(self):
        x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX = self.d
        assert_almost_equal(mX.var(axis=None), mX.compressed().var())
        assert_almost_equal(mX.std(axis=None), mX.compressed().std())
        assert_almost_equal(mX.std(axis=None, ddof=1), mX.compressed().std(ddof=1))
        assert_almost_equal(mX.var(axis=None, ddof=1), mX.compressed().var(ddof=1))
        assert_equal(mXX.var(axis=3).shape, XX.var(axis=3).shape)
        assert_equal(mX.var().shape, X.var().shape)
        mXvar0, mXvar1 = (mX.var(axis=0), mX.var(axis=1))
        assert_almost_equal(mX.var(axis=None, ddof=2), mX.compressed().var(ddof=2))
        assert_almost_equal(mX.std(axis=None, ddof=2), mX.compressed().std(ddof=2))
        for k in range(6):
            assert_almost_equal(mXvar1[k], mX[k].compressed().var())
            assert_almost_equal(mXvar0[k], mX[:, k].compressed().var())
            assert_almost_equal(np.sqrt(mXvar0[k]), mX[:, k].compressed().std())

    @suppress_copy_mask_on_assignment
    def test_varstd_specialcases(self):
        nout = np.array(-1, dtype=float)
        mout = array(-1, dtype=float)
        x = array(arange(10), mask=True)
        for methodname in ('var', 'std'):
            method = getattr(x, methodname)
            assert_(method() is masked)
            assert_(method(0) is masked)
            assert_(method(-1) is masked)
            method(out=mout)
            assert_(mout is not masked)
            assert_equal(mout.mask, True)
            method(out=nout)
            assert_(np.isnan(nout))
        x = array(arange(10), mask=True)
        x[-1] = 9
        for methodname in ('var', 'std'):
            method = getattr(x, methodname)
            assert_(method(ddof=1) is masked)
            assert_(method(0, ddof=1) is masked)
            assert_(method(-1, ddof=1) is masked)
            method(out=mout, ddof=1)
            assert_(mout is not masked)
            assert_equal(mout.mask, True)
            method(out=nout, ddof=1)
            assert_(np.isnan(nout))

    def test_varstd_ddof(self):
        a = array([[1, 1, 0], [1, 1, 0]], mask=[[0, 0, 1], [0, 0, 1]])
        test = a.std(axis=0, ddof=0)
        assert_equal(test.filled(0), [0, 0, 0])
        assert_equal(test.mask, [0, 0, 1])
        test = a.std(axis=0, ddof=1)
        assert_equal(test.filled(0), [0, 0, 0])
        assert_equal(test.mask, [0, 0, 1])
        test = a.std(axis=0, ddof=2)
        assert_equal(test.filled(0), [0, 0, 0])
        assert_equal(test.mask, [1, 1, 1])

    def test_diag(self):
        x = arange(9).reshape((3, 3))
        x[1, 1] = masked
        out = np.diag(x)
        assert_equal(out, [0, 4, 8])
        out = diag(x)
        assert_equal(out, [0, 4, 8])
        assert_equal(out.mask, [0, 1, 0])
        out = diag(out)
        control = array([[0, 0, 0], [0, 4, 0], [0, 0, 8]], mask=[[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert_equal(out, control)

    def test_axis_methods_nomask(self):
        a = array([[1, 2, 3], [4, 5, 6]])
        assert_equal(a.sum(0), [5, 7, 9])
        assert_equal(a.sum(-1), [6, 15])
        assert_equal(a.sum(1), [6, 15])
        assert_equal(a.prod(0), [4, 10, 18])
        assert_equal(a.prod(-1), [6, 120])
        assert_equal(a.prod(1), [6, 120])
        assert_equal(a.min(0), [1, 2, 3])
        assert_equal(a.min(-1), [1, 4])
        assert_equal(a.min(1), [1, 4])
        assert_equal(a.max(0), [4, 5, 6])
        assert_equal(a.max(-1), [3, 6])
        assert_equal(a.max(1), [3, 6])

    @requires_memory(free_bytes=2 * 10000 * 1000 * 2)
    def test_mean_overflow(self):
        a = masked_array(np.full((10000, 10000), 65535, dtype=np.uint16), mask=np.zeros((10000, 10000)))
        assert_equal(a.mean(), 65535.0)

    def test_diff_with_prepend(self):
        x = np.array([1, 2, 2, 3, 4, 2, 1, 1])
        a = np.ma.masked_equal(x[3:], value=2)
        a_prep = np.ma.masked_equal(x[:3], value=2)
        diff1 = np.ma.diff(a, prepend=a_prep, axis=0)
        b = np.ma.masked_equal(x, value=2)
        diff2 = np.ma.diff(b, axis=0)
        assert_(np.ma.allequal(diff1, diff2))

    def test_diff_with_append(self):
        x = np.array([1, 2, 2, 3, 4, 2, 1, 1])
        a = np.ma.masked_equal(x[:3], value=2)
        a_app = np.ma.masked_equal(x[3:], value=2)
        diff1 = np.ma.diff(a, append=a_app, axis=0)
        b = np.ma.masked_equal(x, value=2)
        diff2 = np.ma.diff(b, axis=0)
        assert_(np.ma.allequal(diff1, diff2))

    def test_diff_with_dim_0(self):
        with pytest.raises(ValueError, match='diff requires input that is at least one dimensional'):
            np.ma.diff(np.array(1))

    def test_diff_with_n_0(self):
        a = np.ma.masked_equal([1, 2, 2, 3, 4, 2, 1, 1], value=2)
        diff = np.ma.diff(a, n=0, axis=0)
        assert_(np.ma.allequal(a, diff))