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
class TestYeojohnsonNormmax:

    def setup_method(self):
        self.x = _old_loggamma_rvs(5, size=50, random_state=12345) + 5

    def test_mle(self):
        maxlog = stats.yeojohnson_normmax(self.x)
        assert_allclose(maxlog, 1.876393, rtol=1e-06)

    def test_darwin_example(self):
        x = [6.1, -8.4, 1.0, 2.0, 0.7, 2.9, 3.5, 5.1, 1.8, 3.6, 7.0, 3.0, 9.3, 7.5, -6.0]
        lmbda = stats.yeojohnson_normmax(x)
        assert np.allclose(lmbda, 1.305, atol=0.001)