import warnings
import sys
from functools import partial
import numpy as np
from numpy.random import RandomState
from numpy.testing import (assert_array_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import re
from scipy import optimize, stats, special
from scipy.stats._morestats import _abw_state, _get_As_weibull, _Avals_weibull
from .common_tests import check_named_results
from .._hypotests import _get_wilcoxon_distr, _get_wilcoxon_distr2
from scipy.stats._binomtest import _binary_search_for_binom_tst
from scipy.stats._distr_params import distcont
class TestFDRControl:

    def test_input_validation(self):
        message = '`ps` must include only numbers between 0 and 1'
        with pytest.raises(ValueError, match=message):
            stats.false_discovery_control([-1, 0.5, 0.7])
        with pytest.raises(ValueError, match=message):
            stats.false_discovery_control([0.5, 0.7, 2])
        with pytest.raises(ValueError, match=message):
            stats.false_discovery_control([0.5, 0.7, np.nan])
        message = "Unrecognized `method` 'YAK'"
        with pytest.raises(ValueError, match=message):
            stats.false_discovery_control([0.5, 0.7, 0.9], method='YAK')
        message = '`axis` must be an integer or `None`'
        with pytest.raises(ValueError, match=message):
            stats.false_discovery_control([0.5, 0.7, 0.9], axis=1.5)
        with pytest.raises(ValueError, match=message):
            stats.false_discovery_control([0.5, 0.7, 0.9], axis=(1, 2))

    def test_against_TileStats(self):
        ps = [0.005, 0.009, 0.019, 0.022, 0.051, 0.101, 0.361, 0.387]
        res = stats.false_discovery_control(ps)
        ref = [0.036, 0.036, 0.044, 0.044, 0.082, 0.135, 0.387, 0.387]
        assert_allclose(res, ref, atol=0.001)

    @pytest.mark.parametrize('case', [([0.24617028, 0.0114003, 0.05652047, 0.06841983, 0.07989886, 0.0184149, 0.17540784, 0.06841983, 0.06841983, 0.25464082], 'bh'), ([0.72102493, 0.03339112, 0.16554665, 0.20039952, 0.23402122, 0.05393666, 0.51376399, 0.20039952, 0.20039952, 0.74583488], 'by')])
    def test_against_R(self, case):
        ref, method = case
        rng = np.random.default_rng(6134137338861652935)
        ps = stats.loguniform.rvs(0.001, 0.5, size=10, random_state=rng)
        ps[3] = ps[7]
        res = stats.false_discovery_control(ps, method=method)
        assert_allclose(res, ref, atol=1e-06)

    def test_axis_None(self):
        rng = np.random.default_rng(6134137338861652935)
        ps = stats.loguniform.rvs(0.001, 0.5, size=(3, 4, 5), random_state=rng)
        res = stats.false_discovery_control(ps, axis=None)
        ref = stats.false_discovery_control(ps.ravel())
        assert_equal(res, ref)

    @pytest.mark.parametrize('axis', [0, 1, -1])
    def test_axis(self, axis):
        rng = np.random.default_rng(6134137338861652935)
        ps = stats.loguniform.rvs(0.001, 0.5, size=(3, 4, 5), random_state=rng)
        res = stats.false_discovery_control(ps, axis=axis)
        ref = np.apply_along_axis(stats.false_discovery_control, axis, ps)
        assert_equal(res, ref)

    def test_edge_cases(self):
        assert_array_equal(stats.false_discovery_control([0.25]), [0.25])
        assert_array_equal(stats.false_discovery_control(0.25), 0.25)
        assert_array_equal(stats.false_discovery_control([]), [])