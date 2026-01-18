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
class TestYeojohnson_llf:

    def test_array_like(self):
        x = stats.norm.rvs(size=100, loc=0, random_state=54321)
        lmbda = 1
        llf = stats.yeojohnson_llf(lmbda, x)
        llf2 = stats.yeojohnson_llf(lmbda, list(x))
        assert_allclose(llf, llf2, rtol=1e-12)

    def test_2d_input(self):
        x = stats.norm.rvs(size=100, loc=10, random_state=54321)
        lmbda = 1
        llf = stats.yeojohnson_llf(lmbda, x)
        llf2 = stats.yeojohnson_llf(lmbda, np.vstack([x, x]).T)
        assert_allclose([llf, llf], llf2, rtol=1e-12)

    def test_empty(self):
        assert_(np.isnan(stats.yeojohnson_llf(1, [])))