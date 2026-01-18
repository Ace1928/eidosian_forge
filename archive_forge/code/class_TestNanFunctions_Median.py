import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
class TestNanFunctions_Median:

    def test_mutation(self):
        ndat = _ndat.copy()
        np.nanmedian(ndat)
        assert_equal(ndat, _ndat)

    def test_keepdims(self):
        mat = np.eye(3)
        for axis in [None, 0, 1]:
            tgt = np.median(mat, axis=axis, out=None, overwrite_input=False)
            res = np.nanmedian(mat, axis=axis, out=None, overwrite_input=False)
            assert_(res.ndim == tgt.ndim)
        d = np.ones((3, 5, 7, 11))
        w = np.random.random((4, 200)) * np.array(d.shape)[:, None]
        w = w.astype(np.intp)
        d[tuple(w)] = np.nan
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            res = np.nanmedian(d, axis=None, keepdims=True)
            assert_equal(res.shape, (1, 1, 1, 1))
            res = np.nanmedian(d, axis=(0, 1), keepdims=True)
            assert_equal(res.shape, (1, 1, 7, 11))
            res = np.nanmedian(d, axis=(0, 3), keepdims=True)
            assert_equal(res.shape, (1, 5, 7, 1))
            res = np.nanmedian(d, axis=(1,), keepdims=True)
            assert_equal(res.shape, (3, 1, 7, 11))
            res = np.nanmedian(d, axis=(0, 1, 2, 3), keepdims=True)
            assert_equal(res.shape, (1, 1, 1, 1))
            res = np.nanmedian(d, axis=(0, 1, 3), keepdims=True)
            assert_equal(res.shape, (1, 1, 7, 1))

    @pytest.mark.parametrize(argnames='axis', argvalues=[None, 1, (1,), (0, 1), (-3, -1)])
    @pytest.mark.filterwarnings('ignore:All-NaN slice:RuntimeWarning')
    def test_keepdims_out(self, axis):
        d = np.ones((3, 5, 7, 11))
        w = np.random.random((4, 200)) * np.array(d.shape)[:, None]
        w = w.astype(np.intp)
        d[tuple(w)] = np.nan
        if axis is None:
            shape_out = (1,) * d.ndim
        else:
            axis_norm = normalize_axis_tuple(axis, d.ndim)
            shape_out = tuple((1 if i in axis_norm else d.shape[i] for i in range(d.ndim)))
        out = np.empty(shape_out)
        result = np.nanmedian(d, axis=axis, keepdims=True, out=out)
        assert result is out
        assert_equal(result.shape, shape_out)

    def test_out(self):
        mat = np.random.rand(3, 3)
        nan_mat = np.insert(mat, [0, 2], np.nan, axis=1)
        resout = np.zeros(3)
        tgt = np.median(mat, axis=1)
        res = np.nanmedian(nan_mat, axis=1, out=resout)
        assert_almost_equal(res, resout)
        assert_almost_equal(res, tgt)
        resout = np.zeros(())
        tgt = np.median(mat, axis=None)
        res = np.nanmedian(nan_mat, axis=None, out=resout)
        assert_almost_equal(res, resout)
        assert_almost_equal(res, tgt)
        res = np.nanmedian(nan_mat, axis=(0, 1), out=resout)
        assert_almost_equal(res, resout)
        assert_almost_equal(res, tgt)

    def test_small_large(self):
        for s in [5, 20, 51, 200, 1000]:
            d = np.random.randn(4, s)
            w = np.random.randint(0, d.size, size=d.size // 5)
            d.ravel()[w] = np.nan
            d[:, 0] = 1.0
            tgt = []
            for x in d:
                nonan = np.compress(~np.isnan(x), x)
                tgt.append(np.median(nonan, overwrite_input=True))
            assert_array_equal(np.nanmedian(d, axis=-1), tgt)

    def test_result_values(self):
        tgt = [np.median(d) for d in _rdat]
        res = np.nanmedian(_ndat, axis=1)
        assert_almost_equal(res, tgt)

    @pytest.mark.parametrize('axis', [None, 0, 1])
    @pytest.mark.parametrize('dtype', _TYPE_CODES)
    def test_allnans(self, dtype, axis):
        mat = np.full((3, 3), np.nan).astype(dtype)
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning)
            output = np.nanmedian(mat, axis=axis)
            assert output.dtype == mat.dtype
            assert np.isnan(output).all()
            if axis is None:
                assert_(len(sup.log) == 1)
            else:
                assert_(len(sup.log) == 3)
            scalar = np.array(np.nan).astype(dtype)[()]
            output_scalar = np.nanmedian(scalar)
            assert output_scalar.dtype == scalar.dtype
            assert np.isnan(output_scalar)
            if axis is None:
                assert_(len(sup.log) == 2)
            else:
                assert_(len(sup.log) == 4)

    def test_empty(self):
        mat = np.zeros((0, 3))
        for axis in [0, None]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                assert_(np.isnan(np.nanmedian(mat, axis=axis)).all())
                assert_(len(w) == 1)
                assert_(issubclass(w[0].category, RuntimeWarning))
        for axis in [1]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                assert_equal(np.nanmedian(mat, axis=axis), np.zeros([]))
                assert_(len(w) == 0)

    def test_scalar(self):
        assert_(np.nanmedian(0.0) == 0.0)

    def test_extended_axis_invalid(self):
        d = np.ones((3, 5, 7, 11))
        assert_raises(np.AxisError, np.nanmedian, d, axis=-5)
        assert_raises(np.AxisError, np.nanmedian, d, axis=(0, -5))
        assert_raises(np.AxisError, np.nanmedian, d, axis=4)
        assert_raises(np.AxisError, np.nanmedian, d, axis=(0, 4))
        assert_raises(ValueError, np.nanmedian, d, axis=(1, 1))

    def test_float_special(self):
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            for inf in [np.inf, -np.inf]:
                a = np.array([[inf, np.nan], [np.nan, np.nan]])
                assert_equal(np.nanmedian(a, axis=0), [inf, np.nan])
                assert_equal(np.nanmedian(a, axis=1), [inf, np.nan])
                assert_equal(np.nanmedian(a), inf)
                a = np.array([[np.nan, np.nan, inf], [np.nan, np.nan, inf]])
                assert_equal(np.nanmedian(a), inf)
                assert_equal(np.nanmedian(a, axis=0), [np.nan, np.nan, inf])
                assert_equal(np.nanmedian(a, axis=1), inf)
                a = np.array([[inf, inf], [inf, inf]])
                assert_equal(np.nanmedian(a, axis=1), inf)
                a = np.array([[inf, 7, -inf, -9], [-10, np.nan, np.nan, 5], [4, np.nan, np.nan, inf]], dtype=np.float32)
                if inf > 0:
                    assert_equal(np.nanmedian(a, axis=0), [4.0, 7.0, -inf, 5.0])
                    assert_equal(np.nanmedian(a), 4.5)
                else:
                    assert_equal(np.nanmedian(a, axis=0), [-10.0, 7.0, -inf, -9.0])
                    assert_equal(np.nanmedian(a), -2.5)
                assert_equal(np.nanmedian(a, axis=-1), [-1.0, -2.5, inf])
                for i in range(0, 10):
                    for j in range(1, 10):
                        a = np.array([[np.nan] * i + [inf] * j] * 2)
                        assert_equal(np.nanmedian(a), inf)
                        assert_equal(np.nanmedian(a, axis=1), inf)
                        assert_equal(np.nanmedian(a, axis=0), [np.nan] * i + [inf] * j)
                        a = np.array([[np.nan] * i + [-inf] * j] * 2)
                        assert_equal(np.nanmedian(a), -inf)
                        assert_equal(np.nanmedian(a, axis=1), -inf)
                        assert_equal(np.nanmedian(a, axis=0), [np.nan] * i + [-inf] * j)