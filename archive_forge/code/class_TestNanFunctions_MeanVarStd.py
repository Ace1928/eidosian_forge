import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
class TestNanFunctions_MeanVarStd(SharedNanFunctionsTestsMixin):
    nanfuncs = [np.nanmean, np.nanvar, np.nanstd]
    stdfuncs = [np.mean, np.var, np.std]

    def test_dtype_error(self):
        for f in self.nanfuncs:
            for dtype in [np.bool_, np.int_, np.object_]:
                assert_raises(TypeError, f, _ndat, axis=1, dtype=dtype)

    def test_out_dtype_error(self):
        for f in self.nanfuncs:
            for dtype in [np.bool_, np.int_, np.object_]:
                out = np.empty(_ndat.shape[0], dtype=dtype)
                assert_raises(TypeError, f, _ndat, axis=1, out=out)

    def test_ddof(self):
        nanfuncs = [np.nanvar, np.nanstd]
        stdfuncs = [np.var, np.std]
        for nf, rf in zip(nanfuncs, stdfuncs):
            for ddof in [0, 1]:
                tgt = [rf(d, ddof=ddof) for d in _rdat]
                res = nf(_ndat, axis=1, ddof=ddof)
                assert_almost_equal(res, tgt)

    def test_ddof_too_big(self):
        nanfuncs = [np.nanvar, np.nanstd]
        stdfuncs = [np.var, np.std]
        dsize = [len(d) for d in _rdat]
        for nf, rf in zip(nanfuncs, stdfuncs):
            for ddof in range(5):
                with suppress_warnings() as sup:
                    sup.record(RuntimeWarning)
                    sup.filter(np.ComplexWarning)
                    tgt = [ddof >= d for d in dsize]
                    res = nf(_ndat, axis=1, ddof=ddof)
                    assert_equal(np.isnan(res), tgt)
                    if any(tgt):
                        assert_(len(sup.log) == 1)
                    else:
                        assert_(len(sup.log) == 0)

    @pytest.mark.parametrize('axis', [None, 0, 1])
    @pytest.mark.parametrize('dtype', np.typecodes['AllFloat'])
    @pytest.mark.parametrize('array', [np.array(np.nan), np.full((3, 3), np.nan)], ids=['0d', '2d'])
    def test_allnans(self, axis, dtype, array):
        if axis is not None and array.ndim == 0:
            pytest.skip(f'`axis != None` not supported for 0d arrays')
        array = array.astype(dtype)
        match = '(Degrees of freedom <= 0 for slice.)|(Mean of empty slice)'
        for func in self.nanfuncs:
            with pytest.warns(RuntimeWarning, match=match):
                out = func(array, axis=axis)
            assert np.isnan(out).all()
            if func is np.nanmean:
                assert out.dtype == array.dtype
            else:
                assert out.dtype == np.abs(array).dtype

    def test_empty(self):
        mat = np.zeros((0, 3))
        for f in self.nanfuncs:
            for axis in [0, None]:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    assert_(np.isnan(f(mat, axis=axis)).all())
                    assert_(len(w) == 1)
                    assert_(issubclass(w[0].category, RuntimeWarning))
            for axis in [1]:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    assert_equal(f(mat, axis=axis), np.zeros([]))
                    assert_(len(w) == 0)

    @pytest.mark.parametrize('dtype', np.typecodes['AllFloat'])
    def test_where(self, dtype):
        ar = np.arange(9).reshape(3, 3).astype(dtype)
        ar[0, :] = np.nan
        where = np.ones_like(ar, dtype=np.bool_)
        where[:, 0] = False
        for f, f_std in zip(self.nanfuncs, self.stdfuncs):
            reference = f_std(ar[where][2:])
            dtype_reference = dtype if f is np.nanmean else ar.real.dtype
            ret = f(ar, where=where)
            assert ret.dtype == dtype_reference
            np.testing.assert_allclose(ret, reference)