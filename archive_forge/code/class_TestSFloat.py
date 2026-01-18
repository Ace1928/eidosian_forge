import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
class TestSFloat:

    def _get_array(self, scaling, aligned=True):
        if not aligned:
            a = np.empty(3 * 8 + 1, dtype=np.uint8)[1:]
            a = a.view(np.float64)
            a[:] = [1.0, 2.0, 3.0]
        else:
            a = np.array([1.0, 2.0, 3.0])
        a *= 1.0 / scaling
        return a.view(SF(scaling))

    def test_sfloat_rescaled(self):
        sf = SF(1.0)
        sf2 = sf.scaled_by(2.0)
        assert sf2.get_scaling() == 2.0
        sf6 = sf2.scaled_by(3.0)
        assert sf6.get_scaling() == 6.0

    def test_class_discovery(self):
        dt, _ = discover_array_params([1.0, 2.0, 3.0], dtype=SF)
        assert dt == SF(1.0)

    @pytest.mark.parametrize('scaling', [1.0, -1.0, 2.0])
    def test_scaled_float_from_floats(self, scaling):
        a = np.array([1.0, 2.0, 3.0], dtype=SF(scaling))
        assert a.dtype.get_scaling() == scaling
        assert_array_equal(scaling * a.view(np.float64), [1.0, 2.0, 3.0])

    def test_repr(self):
        assert repr(SF(scaling=1.0)) == '_ScaledFloatTestDType(scaling=1.0)'

    def test_dtype_name(self):
        assert SF(1.0).name == '_ScaledFloatTestDType64'

    @pytest.mark.parametrize('scaling', [1.0, -1.0, 2.0])
    def test_sfloat_from_float(self, scaling):
        a = np.array([1.0, 2.0, 3.0]).astype(dtype=SF(scaling))
        assert a.dtype.get_scaling() == scaling
        assert_array_equal(scaling * a.view(np.float64), [1.0, 2.0, 3.0])

    @pytest.mark.parametrize('aligned', [True, False])
    @pytest.mark.parametrize('scaling', [1.0, -1.0, 2.0])
    def test_sfloat_getitem(self, aligned, scaling):
        a = self._get_array(1.0, aligned)
        assert a.tolist() == [1.0, 2.0, 3.0]

    @pytest.mark.parametrize('aligned', [True, False])
    def test_sfloat_casts(self, aligned):
        a = self._get_array(1.0, aligned)
        assert np.can_cast(a, SF(-1.0), casting='equiv')
        assert not np.can_cast(a, SF(-1.0), casting='no')
        na = a.astype(SF(-1.0))
        assert_array_equal(-1 * na.view(np.float64), a.view(np.float64))
        assert np.can_cast(a, SF(2.0), casting='same_kind')
        assert not np.can_cast(a, SF(2.0), casting='safe')
        a2 = a.astype(SF(2.0))
        assert_array_equal(2 * a2.view(np.float64), a.view(np.float64))

    @pytest.mark.parametrize('aligned', [True, False])
    def test_sfloat_cast_internal_errors(self, aligned):
        a = self._get_array(2e+300, aligned)
        with pytest.raises(TypeError, match='error raised inside the core-loop: non-finite factor!'):
            a.astype(SF(2e-300))

    def test_sfloat_promotion(self):
        assert np.result_type(SF(2.0), SF(3.0)) == SF(3.0)
        assert np.result_type(SF(3.0), SF(2.0)) == SF(3.0)
        assert np.result_type(SF(3.0), np.float64) == SF(3.0)
        assert np.result_type(np.float64, SF(0.5)) == SF(1.0)
        with pytest.raises(TypeError):
            np.result_type(SF(1.0), np.int64)

    def test_basic_multiply(self):
        a = self._get_array(2.0)
        b = self._get_array(4.0)
        res = a * b
        assert res.dtype.get_scaling() == 8.0
        expected_view = a.view(np.float64) * b.view(np.float64)
        assert_array_equal(res.view(np.float64), expected_view)

    def test_possible_and_impossible_reduce(self):
        a = self._get_array(2.0)
        res = np.add.reduce(a, initial=0.0)
        assert res == a.astype(np.float64).sum()
        with pytest.raises(TypeError, match='the resolved dtypes are not compatible'):
            np.multiply.reduce(a)

    def test_basic_ufunc_at(self):
        float_a = np.array([1.0, 2.0, 3.0])
        b = self._get_array(2.0)
        float_b = b.view(np.float64).copy()
        np.multiply.at(float_b, [1, 1, 1], float_a)
        np.multiply.at(b, [1, 1, 1], float_a)
        assert_array_equal(b.view(np.float64), float_b)

    def test_basic_multiply_promotion(self):
        float_a = np.array([1.0, 2.0, 3.0])
        b = self._get_array(2.0)
        res1 = float_a * b
        res2 = b * float_a
        assert res1.dtype == res2.dtype == b.dtype
        expected_view = float_a * b.view(np.float64)
        assert_array_equal(res1.view(np.float64), expected_view)
        assert_array_equal(res2.view(np.float64), expected_view)
        np.multiply(b, float_a, out=res2)
        with pytest.raises(TypeError):
            np.multiply(b, float_a, out=np.arange(3))

    def test_basic_addition(self):
        a = self._get_array(2.0)
        b = self._get_array(4.0)
        res = a + b
        assert res.dtype == np.result_type(a.dtype, b.dtype)
        expected_view = a.astype(res.dtype).view(np.float64) + b.astype(res.dtype).view(np.float64)
        assert_array_equal(res.view(np.float64), expected_view)

    def test_addition_cast_safety(self):
        """The addition method is special for the scaled float, because it
        includes the "cast" between different factors, thus cast-safety
        is influenced by the implementation.
        """
        a = self._get_array(2.0)
        b = self._get_array(-2.0)
        c = self._get_array(3.0)
        np.add(a, b, casting='equiv')
        with pytest.raises(TypeError):
            np.add(a, b, casting='no')
        with pytest.raises(TypeError):
            np.add(a, c, casting='safe')
        with pytest.raises(TypeError):
            np.add(a, a, out=c, casting='safe')

    @pytest.mark.parametrize('ufunc', [np.logical_and, np.logical_or, np.logical_xor])
    def test_logical_ufuncs_casts_to_bool(self, ufunc):
        a = self._get_array(2.0)
        a[0] = 0.0
        float_equiv = a.astype(float)
        expected = ufunc(float_equiv, float_equiv)
        res = ufunc(a, a)
        assert_array_equal(res, expected)
        expected = ufunc.reduce(float_equiv)
        res = ufunc.reduce(a)
        assert_array_equal(res, expected)
        with pytest.raises(TypeError):
            ufunc(a, a, out=np.empty(a.shape, dtype=int), casting='equiv')

    def test_wrapped_and_wrapped_reductions(self):
        a = self._get_array(2.0)
        float_equiv = a.astype(float)
        expected = np.hypot(float_equiv, float_equiv)
        res = np.hypot(a, a)
        assert res.dtype == a.dtype
        res_float = res.view(np.float64) * 2
        assert_array_equal(res_float, expected)
        res = np.hypot.reduce(a, keepdims=True)
        assert res.dtype == a.dtype
        expected = np.hypot.reduce(float_equiv, keepdims=True)
        assert res.view(np.float64) * 2 == expected

    def test_astype_class(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=object)
        res = arr.astype(SF)
        expected = arr.astype(SF(1.0))
        assert_array_equal(res.view(np.float64), expected.view(np.float64))

    def test_creation_class(self):
        arr1 = np.array([1.0, 2.0, 3.0], dtype=SF)
        assert arr1.dtype == SF(1.0)
        arr2 = np.array([1.0, 2.0, 3.0], dtype=SF(1.0))
        assert_array_equal(arr1.view(np.float64), arr2.view(np.float64))