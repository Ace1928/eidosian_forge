import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from itertools import product
from math import gcd
import pytest
from pytest import raises as assert_raises
from numpy.testing import (
from numpy import array, arange
import numpy as np
from scipy.fft import fft
from scipy.ndimage import correlate1d
from scipy.optimize import fmin, linear_sum_assignment
from scipy import signal
from scipy.signal import (
from scipy.signal.windows import hann
from scipy.signal._signaltools import (_filtfilt_gust, _compute_factors,
from scipy.signal._upfirdn import _upfirdn_modes
from scipy._lib import _testutils
from scipy._lib._util import ComplexWarning, np_long, np_ulong
class TestUniqueRoots:

    def test_real_no_repeat(self):
        p = [-1.0, -0.5, 0.3, 1.2, 10.0]
        unique, multiplicity = unique_roots(p)
        assert_almost_equal(unique, p, decimal=15)
        assert_equal(multiplicity, np.ones(len(p)))

    def test_real_repeat(self):
        p = [-1.0, -0.95, -0.89, -0.8, 0.5, 1.0, 1.05]
        unique, multiplicity = unique_roots(p, tol=0.1, rtype='min')
        assert_almost_equal(unique, [-1.0, -0.89, 0.5, 1.0], decimal=15)
        assert_equal(multiplicity, [2, 2, 1, 2])
        unique, multiplicity = unique_roots(p, tol=0.1, rtype='max')
        assert_almost_equal(unique, [-0.95, -0.8, 0.5, 1.05], decimal=15)
        assert_equal(multiplicity, [2, 2, 1, 2])
        unique, multiplicity = unique_roots(p, tol=0.1, rtype='avg')
        assert_almost_equal(unique, [-0.975, -0.845, 0.5, 1.025], decimal=15)
        assert_equal(multiplicity, [2, 2, 1, 2])

    def test_complex_no_repeat(self):
        p = [-1.0, 1j, 0.5 + 0.5j, -1.0 - 1j, 3.0 + 2j]
        unique, multiplicity = unique_roots(p)
        assert_almost_equal(unique, p, decimal=15)
        assert_equal(multiplicity, np.ones(len(p)))

    def test_complex_repeat(self):
        p = [-1.0, -1.0 + 0.05j, -0.95 + 0.15j, -0.9 + 0.15j, 0.0, 0.5 + 0.5j, 0.45 + 0.55j]
        unique, multiplicity = unique_roots(p, tol=0.1, rtype='min')
        assert_almost_equal(unique, [-1.0, -0.95 + 0.15j, 0.0, 0.45 + 0.55j], decimal=15)
        assert_equal(multiplicity, [2, 2, 1, 2])
        unique, multiplicity = unique_roots(p, tol=0.1, rtype='max')
        assert_almost_equal(unique, [-1.0 + 0.05j, -0.9 + 0.15j, 0.0, 0.5 + 0.5j], decimal=15)
        assert_equal(multiplicity, [2, 2, 1, 2])
        unique, multiplicity = unique_roots(p, tol=0.1, rtype='avg')
        assert_almost_equal(unique, [-1.0 + 0.025j, -0.925 + 0.15j, 0.0, 0.475 + 0.525j], decimal=15)
        assert_equal(multiplicity, [2, 2, 1, 2])

    def test_gh_4915(self):
        p = np.roots(np.convolve(np.ones(5), np.ones(5)))
        true_roots = [-(-1) ** (1 / 5), (-1) ** (4 / 5), -(-1) ** (3 / 5), (-1) ** (2 / 5)]
        unique, multiplicity = unique_roots(p)
        unique = np.sort(unique)
        assert_almost_equal(np.sort(unique), true_roots, decimal=7)
        assert_equal(multiplicity, [2, 2, 2, 2])

    def test_complex_roots_extra(self):
        unique, multiplicity = unique_roots([1.0, 1j, 1.0])
        assert_almost_equal(unique, [1.0, 1j], decimal=15)
        assert_equal(multiplicity, [2, 1])
        unique, multiplicity = unique_roots([1, 1 + 2e-09, 1e-09 + 1j], tol=0.1)
        assert_almost_equal(unique, [1.0, 1e-09 + 1j], decimal=15)
        assert_equal(multiplicity, [2, 1])

    def test_single_unique_root(self):
        p = np.random.rand(100) + 1j * np.random.rand(100)
        unique, multiplicity = unique_roots(p, 2)
        assert_almost_equal(unique, [np.min(p)], decimal=15)
        assert_equal(multiplicity, [100])