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
class TestStirling2:
    table = [[1], [0, 1], [0, 1, 1], [0, 1, 3, 1], [0, 1, 7, 6, 1], [0, 1, 15, 25, 10, 1], [0, 1, 31, 90, 65, 15, 1], [0, 1, 63, 301, 350, 140, 21, 1], [0, 1, 127, 966, 1701, 1050, 266, 28, 1], [0, 1, 255, 3025, 7770, 6951, 2646, 462, 36, 1], [0, 1, 511, 9330, 34105, 42525, 22827, 5880, 750, 45, 1]]

    @pytest.mark.parametrize('is_exact, comp, kwargs', [(True, assert_equal, {}), (False, assert_allclose, {'rtol': 1e-12})])
    def test_table_cases(self, is_exact, comp, kwargs):
        for n in range(1, len(self.table)):
            k_values = list(range(n + 1))
            row = self.table[n]
            comp(row, stirling2([n], k_values, exact=is_exact), **kwargs)

    @pytest.mark.parametrize('is_exact, comp, kwargs', [(True, assert_equal, {}), (False, assert_allclose, {'rtol': 1e-12})])
    def test_valid_single_integer(self, is_exact, comp, kwargs):
        comp(stirling2(0, 0, exact=is_exact), self.table[0][0], **kwargs)
        comp(stirling2(4, 2, exact=is_exact), self.table[4][2], **kwargs)
        comp(stirling2(5, 3, exact=is_exact), 25, **kwargs)
        comp(stirling2([5], [3], exact=is_exact), [25], **kwargs)

    @pytest.mark.parametrize('is_exact, comp, kwargs', [(True, assert_equal, {}), (False, assert_allclose, {'rtol': 1e-12})])
    def test_negative_integer(self, is_exact, comp, kwargs):
        comp(stirling2(-1, -1, exact=is_exact), 0, **kwargs)
        comp(stirling2(-1, 2, exact=is_exact), 0, **kwargs)
        comp(stirling2(2, -1, exact=is_exact), 0, **kwargs)

    @pytest.mark.parametrize('is_exact, comp, kwargs', [(True, assert_equal, {}), (False, assert_allclose, {'rtol': 1e-12})])
    def test_array_inputs(self, is_exact, comp, kwargs):
        ans = [self.table[10][3], self.table[10][4]]
        comp(stirling2(asarray([10, 10]), asarray([3, 4]), exact=is_exact), ans)
        comp(stirling2([10, 10], asarray([3, 4]), exact=is_exact), ans)
        comp(stirling2(asarray([10, 10]), [3, 4], exact=is_exact), ans)

    @pytest.mark.parametrize('is_exact, comp, kwargs', [(True, assert_equal, {}), (False, assert_allclose, {'rtol': 1e-13})])
    def test_mixed_values(self, is_exact, comp, kwargs):
        ans = [0, 1, 3, 25, 1050, 5880, 9330]
        n = [-1, 0, 3, 5, 8, 10, 10]
        k = [-2, 0, 2, 3, 5, 7, 3]
        comp(stirling2(n, k, exact=is_exact), ans, **kwargs)

    def test_correct_parity(self):
        """Test parity follows well known identity.

        en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind#Parity
        """
        n, K = (100, np.arange(101))
        assert_equal(stirling2(n, K, exact=True) % 2, [math.comb(n - k // 2 - 1, n - k) % 2 for k in K])

    def test_big_numbers(self):
        ans = asarray([48063331393110, 48004081105038305])
        n = [25, 30]
        k = [17, 4]
        assert array_equal(stirling2(n, k, exact=True), ans)
        ans = asarray([2801934359500572414253157841233849412, 14245032222277144547280648984426251])
        n = [42, 43]
        k = [17, 23]
        assert array_equal(stirling2(n, k, exact=True), ans)

    @pytest.mark.parametrize('N', [4.5, 3.0, 4 + 1j, '12', np.nan])
    @pytest.mark.parametrize('K', [3.5, 3, '2', None])
    @pytest.mark.parametrize('is_exact', [True, False])
    def test_unsupported_input_types(self, N, K, is_exact):
        with pytest.raises(TypeError):
            stirling2(N, K, exact=is_exact)

    @pytest.mark.parametrize('is_exact', [True, False])
    def test_numpy_array_int_object_dtype(self, is_exact):
        ans = asarray(self.table[4][1:])
        n = asarray([4, 4, 4, 4], dtype=object)
        k = asarray([1, 2, 3, 4], dtype=object)
        with pytest.raises(TypeError):
            array_equal(stirling2(n, k, exact=is_exact), ans)

    @pytest.mark.parametrize('is_exact, comp, kwargs', [(True, assert_equal, {}), (False, assert_allclose, {'rtol': 1e-13})])
    def test_numpy_array_unsigned_int_dtype(self, is_exact, comp, kwargs):
        ans = asarray(self.table[4][1:])
        n = asarray([4, 4, 4, 4], dtype=np_ulong)
        k = asarray([1, 2, 3, 4], dtype=np_ulong)
        comp(stirling2(n, k, exact=False), ans, **kwargs)

    @pytest.mark.parametrize('is_exact, comp, kwargs', [(True, assert_equal, {}), (False, assert_allclose, {'rtol': 1e-13})])
    def test_broadcasting_arrays_correctly(self, is_exact, comp, kwargs):
        ans = asarray([[1, 15, 25, 10], [1, 7, 6, 1]])
        n = asarray([[5, 5, 5, 5], [4, 4, 4, 4]])
        k = asarray([1, 2, 3, 4])
        comp(stirling2(n, k, exact=is_exact), ans, **kwargs)
        n = asarray([[4], [4], [4], [4], [4]])
        k = asarray([0, 1, 2, 3, 4, 5])
        ans = asarray([[0, 1, 7, 6, 1, 0] for _ in range(5)])
        comp(stirling2(n, k, exact=False), ans, **kwargs)

    def test_temme_rel_max_error(self):
        x = list(range(51, 101, 5))
        for n in x:
            k_entries = list(range(1, n + 1))
            denom = stirling2([n], k_entries, exact=True)
            num = denom - stirling2([n], k_entries, exact=False)
            assert np.max(np.abs(num / denom)) < 2e-05