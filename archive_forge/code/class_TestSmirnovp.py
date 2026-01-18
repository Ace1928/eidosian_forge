import itertools
import sys
import pytest
import numpy as np
from numpy.testing import assert_
from scipy.special._testutils import FuncData
from scipy.special import kolmogorov, kolmogi, smirnov, smirnovi
from scipy.special._ufuncs import (_kolmogc, _kolmogci, _kolmogp,
class TestSmirnovp:

    def test_nan(self):
        assert_(np.isnan(_smirnovp(1, np.nan)))

    def test_basic(self):
        n1_10 = np.arange(1, 10)
        dataset0 = np.column_stack([n1_10, np.full_like(n1_10, 0), np.full_like(n1_10, -1)])
        FuncData(_smirnovp, dataset0, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        n2_10 = np.arange(2, 10)
        dataset1 = np.column_stack([n2_10, np.full_like(n2_10, 1.0), np.full_like(n2_10, 0)])
        FuncData(_smirnovp, dataset1, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_oneminusoneovern(self):
        n = np.arange(1, 20)
        x = 1.0 / n
        xm1 = 1 - 1.0 / n
        pp1 = -n * x ** (n - 1)
        pp1 -= (1 - np.sign(n - 2) ** 2) * 0.5
        dataset1 = np.column_stack([n, xm1, pp1])
        FuncData(_smirnovp, dataset1, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_oneovertwon(self):
        n = np.arange(1, 20)
        x = 1.0 / 2 / n
        pp = -(n * x + 1) * (1 + x) ** (n - 2)
        dataset0 = np.column_stack([n, x, pp])
        FuncData(_smirnovp, dataset0, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_oneovern(self):
        n = 2 ** np.arange(1, 10)
        x = 1.0 / n
        pp = -(n * x + 1) * (1 + x) ** (n - 2) + 0.5
        dataset0 = np.column_stack([n, x, pp])
        FuncData(_smirnovp, dataset0, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    @pytest.mark.xfail(sys.maxsize <= 2 ** 32, reason='requires 64-bit platform')
    def test_oneovernclose(self):
        n = np.arange(3, 20)
        x = 1.0 / n - 2 * np.finfo(float).eps
        pp = -(n * x + 1) * (1 + x) ** (n - 2)
        dataset0 = np.column_stack([n, x, pp])
        FuncData(_smirnovp, dataset0, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        x = 1.0 / n + 2 * np.finfo(float).eps
        pp = -(n * x + 1) * (1 + x) ** (n - 2) + 1
        dataset1 = np.column_stack([n, x, pp])
        FuncData(_smirnovp, dataset1, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])