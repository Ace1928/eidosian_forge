import os
import re
import warnings
from collections import namedtuple
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib
from numpy.testing import (assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize
from .common_tests import check_named_results
from scipy.spatial.distance import cdist
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
from scipy._lib._util import AxisError
class TestFOneWay:

    def test_trivial(self):
        F, p = stats.f_oneway([0, 2], [0, 2])
        assert_equal(F, 0.0)
        assert_equal(p, 1.0)

    def test_basic(self):
        F, p = stats.f_oneway([0, 2], [2, 4])
        assert_equal(F, 2.0)
        assert_allclose(p, 1 - np.sqrt(0.5), rtol=1e-14)

    def test_known_exact(self):
        F, p = stats.f_oneway([2], [2], [2, 3, 4])
        assert_equal(F, 3 / 5)
        assert_equal(p, 5 / 8)

    def test_large_integer_array(self):
        a = np.array([655, 788], dtype=np.uint16)
        b = np.array([789, 772], dtype=np.uint16)
        F, p = stats.f_oneway(a, b)
        assert_allclose(F, 0.7745021693180554, rtol=1e-14)

    def test_result_attributes(self):
        a = np.array([655, 788], dtype=np.uint16)
        b = np.array([789, 772], dtype=np.uint16)
        res = stats.f_oneway(a, b)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes)

    def test_nist(self):
        filenames = ['SiRstv.dat', 'SmLs01.dat', 'SmLs02.dat', 'SmLs03.dat', 'AtmWtAg.dat', 'SmLs04.dat', 'SmLs05.dat', 'SmLs06.dat', 'SmLs07.dat', 'SmLs08.dat', 'SmLs09.dat']
        for test_case in filenames:
            rtol = 1e-07
            fname = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/nist_anova', test_case))
            with open(fname) as f:
                content = f.read().split('\n')
            certified = [line.split() for line in content[40:48] if line.strip()]
            dataf = np.loadtxt(fname, skiprows=60)
            y, x = dataf.T
            y = y.astype(int)
            caty = np.unique(y)
            f = float(certified[0][-1])
            xlist = [x[y == i] for i in caty]
            res = stats.f_oneway(*xlist)
            hard_tc = ('SmLs07.dat', 'SmLs08.dat', 'SmLs09.dat')
            if test_case in hard_tc:
                rtol = 0.0001
            assert_allclose(res[0], f, rtol=rtol, err_msg='Failing testcase: %s' % test_case)

    @pytest.mark.parametrize('a, b, expected', [(np.array([42, 42, 42]), np.array([7, 7, 7]), (np.inf, 0)), (np.array([42, 42, 42]), np.array([42, 42, 42]), (np.nan, np.nan))])
    def test_constant_input(self, a, b, expected):
        msg = 'Each of the input arrays is constant;'
        with assert_warns(stats.ConstantInputWarning, match=msg):
            f, p = stats.f_oneway(a, b)
            assert f, p == expected

    @pytest.mark.parametrize('axis', [-2, -1, 0, 1])
    def test_2d_inputs(self, axis):
        a = np.array([[1, 4, 3, 3], [2, 5, 3, 3], [3, 6, 3, 3], [2, 3, 3, 3], [1, 4, 3, 3]])
        b = np.array([[3, 1, 5, 3], [4, 6, 5, 3], [4, 3, 5, 3], [1, 5, 5, 3], [5, 5, 5, 3], [2, 3, 5, 3], [8, 2, 5, 3], [2, 2, 5, 3]])
        c = np.array([[4, 3, 4, 3], [4, 2, 4, 3], [5, 4, 4, 3], [5, 4, 4, 3]])
        if axis in [-1, 1]:
            a = a.T
            b = b.T
            c = c.T
            take_axis = 0
        else:
            take_axis = 1
        warn_msg = 'Each of the input arrays is constant;'
        with assert_warns(stats.ConstantInputWarning, match=warn_msg):
            f, p = stats.f_oneway(a, b, c, axis=axis)
        for j in [0, 1]:
            fj, pj = stats.f_oneway(np.take(a, j, take_axis), np.take(b, j, take_axis), np.take(c, j, take_axis))
            assert_allclose(f[j], fj, rtol=1e-14)
            assert_allclose(p[j], pj, rtol=1e-14)
        for j in [2, 3]:
            with assert_warns(stats.ConstantInputWarning, match=warn_msg):
                fj, pj = stats.f_oneway(np.take(a, j, take_axis), np.take(b, j, take_axis), np.take(c, j, take_axis))
                assert_equal(f[j], fj)
                assert_equal(p[j], pj)

    def test_3d_inputs(self):
        a = 1 / np.arange(1.0, 4 * 5 * 7 + 1).reshape(4, 5, 7)
        b = 2 / np.arange(1.0, 4 * 8 * 7 + 1).reshape(4, 8, 7)
        c = np.cos(1 / np.arange(1.0, 4 * 4 * 7 + 1).reshape(4, 4, 7))
        f, p = stats.f_oneway(a, b, c, axis=1)
        assert f.shape == (4, 7)
        assert p.shape == (4, 7)
        for i in range(a.shape[0]):
            for j in range(a.shape[2]):
                fij, pij = stats.f_oneway(a[i, :, j], b[i, :, j], c[i, :, j])
                assert_allclose(fij, f[i, j])
                assert_allclose(pij, p[i, j])

    def test_length0_1d_error(self):
        msg = 'all input arrays have length 1.'
        with assert_warns(stats.DegenerateDataWarning, match=msg):
            result = stats.f_oneway([1, 2, 3], [], [4, 5, 6, 7])
            assert_equal(result, (np.nan, np.nan))

    def test_length0_2d_error(self):
        msg = 'all input arrays have length 1.'
        with assert_warns(stats.DegenerateDataWarning, match=msg):
            ncols = 3
            a = np.ones((4, ncols))
            b = np.ones((0, ncols))
            c = np.ones((5, ncols))
            f, p = stats.f_oneway(a, b, c)
            nans = np.full((ncols,), fill_value=np.nan)
            assert_equal(f, nans)
            assert_equal(p, nans)

    def test_all_length_one(self):
        msg = 'all input arrays have length 1.'
        with assert_warns(stats.DegenerateDataWarning, match=msg):
            result = stats.f_oneway([10], [11], [12], [13])
            assert_equal(result, (np.nan, np.nan))

    @pytest.mark.parametrize('args', [(), ([1, 2, 3],)])
    def test_too_few_inputs(self, args):
        with assert_raises(TypeError):
            stats.f_oneway(*args)

    def test_axis_error(self):
        a = np.ones((3, 4))
        b = np.ones((5, 4))
        with assert_raises(AxisError):
            stats.f_oneway(a, b, axis=2)

    def test_bad_shapes(self):
        a = np.ones((3, 4))
        b = np.ones((5, 4))
        with assert_raises(ValueError):
            stats.f_oneway(a, b, axis=1)