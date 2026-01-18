import functools
import itertools
import operator
import platform
import sys
import numpy as np
from numpy import (array, isnan, r_, arange, finfo, pi, sin, cos, tan, exp,
import pytest
from pytest import raises as assert_raises
from numpy.testing import (assert_equal, assert_almost_equal,
from scipy import special
import scipy.special._ufuncs as cephes
from scipy.special import ellipe, ellipk, ellipkm1
from scipy.special import elliprc, elliprd, elliprf, elliprg, elliprj
from scipy.special import mathieu_odd_coef, mathieu_even_coef, stirling2
from scipy._lib.deprecation import _NoValue
from scipy._lib._util import np_long, np_ulong
from scipy.special._basic import _FACTORIALK_LIMITS_64BITS, \
from scipy.special._testutils import with_special_errors, \
import math
class TestFactorialFunctions:

    @pytest.mark.parametrize('exact', [True, False])
    def test_factorialx_scalar_return_type(self, exact):
        assert np.isscalar(special.factorial(1, exact=exact))
        assert np.isscalar(special.factorial2(1, exact=exact))
        assert np.isscalar(special.factorialk(1, 3, exact=True))

    @pytest.mark.parametrize('n', [-1, -2, -3])
    @pytest.mark.parametrize('exact', [True, False])
    def test_factorialx_negative(self, exact, n):
        assert_equal(special.factorial(n, exact=exact), 0)
        assert_equal(special.factorial2(n, exact=exact), 0)
        assert_equal(special.factorialk(n, 3, exact=True), 0)

    @pytest.mark.parametrize('exact', [True, False])
    def test_factorialx_negative_array(self, exact):
        assert_func = assert_array_equal if exact else assert_allclose
        assert_func(special.factorial([-5, -4, 0, 1], exact=exact), [0, 0, 1, 1])
        assert_func(special.factorial2([-5, -4, 0, 1], exact=exact), [0, 0, 1, 1])
        assert_func(special.factorialk([-5, -4, 0, 1], 3, exact=True), [0, 0, 1, 1])

    @pytest.mark.parametrize('exact', [True, False])
    @pytest.mark.parametrize('content', [np.nan, None, np.datetime64('nat')], ids=['NaN', 'None', 'NaT'])
    def test_factorialx_nan(self, content, exact):
        assert special.factorial(content, exact=exact) is np.nan
        assert special.factorial2(content, exact=exact) is np.nan
        assert special.factorialk(content, 3, exact=True) is np.nan
        if content is not np.nan:
            with pytest.raises(ValueError, match='Unsupported datatype.*'):
                special.factorial([content], exact=exact)
        elif exact:
            with pytest.warns(DeprecationWarning, match='Non-integer array.*'):
                assert np.isnan(special.factorial([content], exact=exact)[0])
        else:
            assert np.isnan(special.factorial([content], exact=exact)[0])
        with pytest.raises(ValueError, match='factorial2 does not support.*'):
            special.factorial2([content], exact=exact)
        with pytest.raises(ValueError, match='factorialk does not support.*'):
            special.factorialk([content], 3, exact=True)

    @pytest.mark.parametrize('levels', range(1, 5))
    @pytest.mark.parametrize('exact', [True, False])
    def test_factorialx_array_shape(self, levels, exact):

        def _nest_me(x, k=1):
            """
            Double x and nest it k times

            For example:
            >>> _nest_me([3, 4], 2)
            [[[3, 4], [3, 4]], [[3, 4], [3, 4]]]
            """
            if k == 0:
                return x
            else:
                return _nest_me([x, x], k - 1)

        def _check(res, nucleus):
            exp = np.array(_nest_me(nucleus, k=levels), dtype=object)
            assert_allclose(res.astype(np.float64), exp.astype(np.float64))
        n = np.array(_nest_me([5, 25], k=levels))
        exp_nucleus = {1: [120, math.factorial(25)], 2: [15, special.factorial2(25, exact=True)], 3: [10, special.factorialk(25, 3)]}
        _check(special.factorial(n, exact=exact), exp_nucleus[1])
        _check(special.factorial2(n, exact=exact), exp_nucleus[2])
        _check(special.factorialk(n, 3, exact=True), exp_nucleus[3])

    @pytest.mark.parametrize('exact', [True, False])
    @pytest.mark.parametrize('dim', range(0, 5))
    def test_factorialx_array_dimension(self, dim, exact):
        n = np.array(5, ndmin=dim)
        exp = {1: 120, 2: 15, 3: 10}
        assert_allclose(special.factorial(n, exact=exact), np.array(exp[1], ndmin=dim))
        assert_allclose(special.factorial2(n, exact=exact), np.array(exp[2], ndmin=dim))
        assert_allclose(special.factorialk(n, 3, exact=True), np.array(exp[3], ndmin=dim))

    @pytest.mark.parametrize('exact', [True, False])
    @pytest.mark.parametrize('level', range(1, 5))
    def test_factorialx_array_like(self, level, exact):

        def _nest_me(x, k=1):
            if k == 0:
                return x
            else:
                return _nest_me([x], k - 1)
        n = _nest_me([5], k=level - 1)
        exp_nucleus = {1: 120, 2: 15, 3: 10}
        assert_func = assert_array_equal if exact else assert_allclose
        assert_func(special.factorial(n, exact=exact), np.array(exp_nucleus[1], ndmin=level))
        assert_func(special.factorial2(n, exact=exact), np.array(exp_nucleus[2], ndmin=level))
        assert_func(special.factorialk(n, 3, exact=True), np.array(exp_nucleus[3], ndmin=level))

    @pytest.mark.parametrize('n', range(30, 180, 10))
    def test_factorial_accuracy(self, n):
        rtol = 6e-14 if sys.platform == 'win32' else 1e-15
        assert_allclose(float(special.factorial(n, exact=True)), special.factorial(n, exact=False), rtol=rtol)
        assert_allclose(special.factorial([n], exact=True).astype(float), special.factorial([n], exact=False), rtol=rtol)

    @pytest.mark.parametrize('n', list(range(0, 22)) + list(range(30, 180, 10)))
    def test_factorial_int_reference(self, n):
        correct = math.factorial(n)
        assert_array_equal(correct, special.factorial(n, True))
        assert_array_equal(correct, special.factorial([n], True)[0])
        rtol = 6e-14 if sys.platform == 'win32' else 1e-15
        assert_allclose(float(correct), special.factorial(n, False), rtol=rtol)
        assert_allclose(float(correct), special.factorial([n], False)[0], rtol=rtol)

    @pytest.mark.parametrize('exact', [True, False])
    def test_factorial_float_reference(self, exact):

        def _check(n, expected):
            assert_allclose(special.factorial(n, exact=exact), expected)
            assert_allclose(special.factorial([n])[0], expected)
        _check(0.01, 0.9943258511915061)
        _check(1.11, 1.051609009483625)
        _check(5.55, 314.9503192327208)
        _check(11.1, 50983227.84411616)
        _check(33.3, 2.4933633396420364e+37)
        _check(55.5, 9.479934358436729e+73)
        _check(77.7, 3.060540559059579e+114)
        _check(99.9, 5.885840419492872e+157)
        _check(170.6243, 1.7969818574957104e+308)

    @pytest.mark.parametrize('dtype', [np.int64, np.float64, np.complex128, object])
    @pytest.mark.parametrize('exact', [True, False])
    @pytest.mark.parametrize('dim', range(0, 5))
    @pytest.mark.parametrize('content', [[], [1], [1.1], [np.nan], [np.nan, 1]], ids=['[]', '[1]', '[1.1]', '[NaN]', '[NaN, 1]'])
    def test_factorial_array_corner_cases(self, content, dim, exact, dtype):
        if dtype == np.int64 and any((np.isnan(x) for x in content)):
            pytest.skip('impossible combination')
        content = content if dim > 0 or len(content) != 1 else content[0]
        n = np.array(content, ndmin=dim, dtype=dtype)
        result = None
        if not content:
            result = special.factorial(n, exact=exact)
        elif not (np.issubdtype(n.dtype, np.integer) or np.issubdtype(n.dtype, np.floating)):
            with pytest.raises(ValueError, match='Unsupported datatype*'):
                special.factorial(n, exact=exact)
        elif exact and (not np.issubdtype(n.dtype, np.integer)) and n.size and np.allclose(n[~np.isnan(n)], n[~np.isnan(n)].astype(np.int64)):
            with pytest.warns(DeprecationWarning, match='Non-integer array.*'):
                result = special.factorial(n, exact=exact)
                if np.any(np.isnan(n)):
                    dtype = np.dtype(np.float64)
                else:
                    dtype = np.dtype(int)
        elif exact and (not np.issubdtype(n.dtype, np.integer)):
            with pytest.raises(ValueError, match='factorial with exact=.*'):
                special.factorial(n, exact=exact)
        else:
            result = special.factorial(n, exact=exact)

        def assert_really_equal(x, y):
            assert type(x) == type(y), f'types not equal: {type(x)}, {type(y)}'
            assert_equal(x, y)
        if result is not None:
            with suppress_warnings() as sup:
                sup.filter(DeprecationWarning)
                n_flat = n.ravel() if n.ndim else n
                r = special.factorial(n_flat, exact=exact) if n.size else []
            expected = np.array(r, ndmin=dim, dtype=dtype)
            assert_really_equal(result, expected)

    @pytest.mark.parametrize('exact', [True, False])
    @pytest.mark.parametrize('n', [1, 1.1, 2 + 2j, np.nan, None], ids=['1', '1.1', '2+2j', 'NaN', 'None'])
    def test_factorial_scalar_corner_cases(self, n, exact):
        if n is None or n is np.nan or np.issubdtype(type(n), np.integer) or np.issubdtype(type(n), np.floating):
            result = special.factorial(n, exact=exact)
            exp = np.nan if n is np.nan or n is None else special.factorial(n)
            assert_equal(result, exp)
        else:
            with pytest.raises(ValueError, match='Unsupported datatype*'):
                special.factorial(n, exact=exact)

    @pytest.mark.parametrize('n', range(30, 180, 11))
    def test_factorial2_accuracy(self, n):
        rtol = 2e-14 if sys.platform == 'win32' else 1e-15
        assert_allclose(float(special.factorial2(n, exact=True)), special.factorial2(n, exact=False), rtol=rtol)
        assert_allclose(special.factorial2([n], exact=True).astype(float), special.factorial2([n], exact=False), rtol=rtol)

    @pytest.mark.parametrize('n', list(range(0, 22)) + list(range(30, 180, 11)))
    def test_factorial2_int_reference(self, n):
        correct = functools.reduce(operator.mul, list(range(n, 0, -2)), 1)
        assert_array_equal(correct, special.factorial2(n, True))
        assert_array_equal(correct, special.factorial2([n], True)[0])
        assert_allclose(float(correct), special.factorial2(n, False))
        assert_allclose(float(correct), special.factorial2([n], False)[0])

    @pytest.mark.parametrize('dtype', [np.int64, np.float64, np.complex128, object])
    @pytest.mark.parametrize('exact', [True, False])
    @pytest.mark.parametrize('dim', range(0, 5))
    @pytest.mark.parametrize('content', [[], [1], [np.nan], [np.nan, 1]], ids=['[]', '[1]', '[NaN]', '[NaN, 1]'])
    def test_factorial2_array_corner_cases(self, content, dim, exact, dtype):
        if dtype == np.int64 and any((np.isnan(x) for x in content)):
            pytest.skip('impossible combination')
        content = content if dim > 0 or len(content) != 1 else content[0]
        n = np.array(content, ndmin=dim, dtype=dtype)
        if np.issubdtype(n.dtype, np.integer) or not content:
            result = special.factorial2(n, exact=exact)
            func = assert_equal if exact or not content else assert_allclose
            func(result, n)
        else:
            with pytest.raises(ValueError, match='factorial2 does not*'):
                special.factorial2(n, 3)

    @pytest.mark.parametrize('exact', [True, False])
    @pytest.mark.parametrize('n', [1, 1.1, 2 + 2j, np.nan, None], ids=['1', '1.1', '2+2j', 'NaN', 'None'])
    def test_factorial2_scalar_corner_cases(self, n, exact):
        if n is None or n is np.nan or np.issubdtype(type(n), np.integer):
            result = special.factorial2(n, exact=exact)
            exp = np.nan if n is np.nan or n is None else special.factorial(n)
            assert_equal(result, exp)
        else:
            with pytest.raises(ValueError, match='factorial2 does not*'):
                special.factorial2(n, exact=exact)

    @pytest.mark.parametrize('k', list(range(1, 5)) + [10, 20])
    @pytest.mark.parametrize('n', list(range(0, 22)) + list(range(22, 100, 11)))
    def test_factorialk_int_reference(self, n, k):
        correct = functools.reduce(operator.mul, list(range(n, 0, -k)), 1)
        assert_array_equal(correct, special.factorialk(n, k, True))
        assert_array_equal(correct, special.factorialk([n], k, True)[0])

    @pytest.mark.parametrize('dtype', [np.int64, np.float64, np.complex128, object])
    @pytest.mark.parametrize('dim', range(0, 5))
    @pytest.mark.parametrize('content', [[], [1], [np.nan], [np.nan, 1]], ids=['[]', '[1]', '[NaN]', '[NaN, 1]'])
    def test_factorialk_array_corner_cases(self, content, dim, dtype):
        if dtype == np.int64 and any((np.isnan(x) for x in content)):
            pytest.skip('impossible combination')
        content = content if dim > 0 or len(content) != 1 else content[0]
        n = np.array(content, ndmin=dim, dtype=dtype)
        if np.issubdtype(n.dtype, np.integer) or not content:
            assert_equal(special.factorialk(n, 3), n)
        else:
            with pytest.raises(ValueError, match='factorialk does not*'):
                special.factorialk(n, 3)

    @pytest.mark.parametrize('exact', [True, False])
    @pytest.mark.parametrize('k', range(1, 5))
    @pytest.mark.parametrize('n', [1, 1.1, 2 + 2j, np.nan, None], ids=['1', '1.1', '2+2j', 'NaN', 'None'])
    def test_factorialk_scalar_corner_cases(self, n, k, exact):
        if not exact:
            with pytest.raises(NotImplementedError):
                special.factorialk(n, k=k, exact=exact)
        elif n is None or n is np.nan or np.issubdtype(type(n), np.integer):
            result = special.factorial2(n, exact=exact)
            nan_cond = n is np.nan or n is None
            expected = np.nan if nan_cond else special.factorialk(n, k=k)
            assert_equal(result, expected)
        else:
            with pytest.raises(ValueError, match='factorialk does not*'):
                special.factorialk(n, k=k, exact=exact)

    @pytest.mark.parametrize('k', [0, 1.1, np.nan, '1'])
    def test_factorialk_raises_k(self, k):
        with pytest.raises(ValueError, match='k must be a positive integer*'):
            special.factorialk(1, k)

    @pytest.mark.parametrize('k', range(1, 12))
    def test_factorialk_dtype(self, k):
        if k in _FACTORIALK_LIMITS_64BITS.keys():
            n = np.array([_FACTORIALK_LIMITS_32BITS[k]])
            assert_equal(special.factorialk(n, k).dtype, np_long)
            assert_equal(special.factorialk(n + 1, k).dtype, np.int64)
            assert special.factorialk(n + 1, k) > np.iinfo(np.int32).max
            n = np.array([_FACTORIALK_LIMITS_64BITS[k]])
            assert_equal(special.factorialk(n, k).dtype, np.int64)
            assert_equal(special.factorialk(n + 1, k).dtype, object)
            assert special.factorialk(n + 1, k) > np.iinfo(np.int64).max
        else:
            assert_equal(special.factorialk(np.array([1]), k).dtype, object)

    def test_factorial_mixed_nan_inputs(self):
        x = np.array([np.nan, 1, 2, 3, np.nan])
        expected = np.array([np.nan, 1, 2, 6, np.nan])
        assert_equal(special.factorial(x, exact=False), expected)
        with pytest.warns(DeprecationWarning, match='Non-integer array.*'):
            assert_equal(special.factorial(x, exact=True), expected)