import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
class TestNanFunctions_MinMax:
    nanfuncs = [np.nanmin, np.nanmax]
    stdfuncs = [np.min, np.max]

    def test_mutation(self):
        ndat = _ndat.copy()
        for f in self.nanfuncs:
            f(ndat)
            assert_equal(ndat, _ndat)

    def test_keepdims(self):
        mat = np.eye(3)
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for axis in [None, 0, 1]:
                tgt = rf(mat, axis=axis, keepdims=True)
                res = nf(mat, axis=axis, keepdims=True)
                assert_(res.ndim == tgt.ndim)

    def test_out(self):
        mat = np.eye(3)
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            resout = np.zeros(3)
            tgt = rf(mat, axis=1)
            res = nf(mat, axis=1, out=resout)
            assert_almost_equal(res, resout)
            assert_almost_equal(res, tgt)

    def test_dtype_from_input(self):
        codes = 'efdgFDG'
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for c in codes:
                mat = np.eye(3, dtype=c)
                tgt = rf(mat, axis=1).dtype.type
                res = nf(mat, axis=1).dtype.type
                assert_(res is tgt)
                tgt = rf(mat, axis=None).dtype.type
                res = nf(mat, axis=None).dtype.type
                assert_(res is tgt)

    def test_result_values(self):
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            tgt = [rf(d) for d in _rdat]
            res = nf(_ndat, axis=1)
            assert_almost_equal(res, tgt)

    @pytest.mark.parametrize('axis', [None, 0, 1])
    @pytest.mark.parametrize('dtype', np.typecodes['AllFloat'])
    @pytest.mark.parametrize('array', [np.array(np.nan), np.full((3, 3), np.nan)], ids=['0d', '2d'])
    def test_allnans(self, axis, dtype, array):
        if axis is not None and array.ndim == 0:
            pytest.skip(f'`axis != None` not supported for 0d arrays')
        array = array.astype(dtype)
        match = 'All-NaN slice encountered'
        for func in self.nanfuncs:
            with pytest.warns(RuntimeWarning, match=match):
                out = func(array, axis=axis)
            assert np.isnan(out).all()
            assert out.dtype == array.dtype

    def test_masked(self):
        mat = np.ma.fix_invalid(_ndat)
        msk = mat._mask.copy()
        for f in [np.nanmin]:
            res = f(mat, axis=1)
            tgt = f(_ndat, axis=1)
            assert_equal(res, tgt)
            assert_equal(mat._mask, msk)
            assert_(not np.isinf(mat).any())

    def test_scalar(self):
        for f in self.nanfuncs:
            assert_(f(0.0) == 0.0)

    def test_subclass(self):

        class MyNDArray(np.ndarray):
            pass
        mine = np.eye(3).view(MyNDArray)
        for f in self.nanfuncs:
            res = f(mine, axis=0)
            assert_(isinstance(res, MyNDArray))
            assert_(res.shape == (3,))
            res = f(mine, axis=1)
            assert_(isinstance(res, MyNDArray))
            assert_(res.shape == (3,))
            res = f(mine)
            assert_(res.shape == ())
        mine[1] = np.nan
        for f in self.nanfuncs:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                res = f(mine, axis=0)
                assert_(isinstance(res, MyNDArray))
                assert_(not np.any(np.isnan(res)))
                assert_(len(w) == 0)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                res = f(mine, axis=1)
                assert_(isinstance(res, MyNDArray))
                assert_(np.isnan(res[1]) and (not np.isnan(res[0])) and (not np.isnan(res[2])))
                assert_(len(w) == 1, 'no warning raised')
                assert_(issubclass(w[0].category, RuntimeWarning))
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                res = f(mine)
                assert_(res.shape == ())
                assert_(res != np.nan)
                assert_(len(w) == 0)

    def test_object_array(self):
        arr = np.array([[1.0, 2.0], [np.nan, 4.0], [np.nan, np.nan]], dtype=object)
        assert_equal(np.nanmin(arr), 1.0)
        assert_equal(np.nanmin(arr, axis=0), [1.0, 2.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            assert_equal(list(np.nanmin(arr, axis=1)), [1.0, 4.0, np.nan])
            assert_(len(w) == 1, 'no warning raised')
            assert_(issubclass(w[0].category, RuntimeWarning))

    @pytest.mark.parametrize('dtype', np.typecodes['AllFloat'])
    def test_initial(self, dtype):

        class MyNDArray(np.ndarray):
            pass
        ar = np.arange(9).astype(dtype)
        ar[:5] = np.nan
        for f in self.nanfuncs:
            initial = 100 if f is np.nanmax else 0
            ret1 = f(ar, initial=initial)
            assert ret1.dtype == dtype
            assert ret1 == initial
            ret2 = f(ar.view(MyNDArray), initial=initial)
            assert ret2.dtype == dtype
            assert ret2 == initial

    @pytest.mark.parametrize('dtype', np.typecodes['AllFloat'])
    def test_where(self, dtype):

        class MyNDArray(np.ndarray):
            pass
        ar = np.arange(9).reshape(3, 3).astype(dtype)
        ar[0, :] = np.nan
        where = np.ones_like(ar, dtype=np.bool_)
        where[:, 0] = False
        for f in self.nanfuncs:
            reference = 4 if f is np.nanmin else 8
            ret1 = f(ar, where=where, initial=5)
            assert ret1.dtype == dtype
            assert ret1 == reference
            ret2 = f(ar.view(MyNDArray), where=where, initial=5)
            assert ret2.dtype == dtype
            assert ret2 == reference