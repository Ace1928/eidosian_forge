import threading
import pickle
import pytest
from copy import deepcopy
import platform
import sys
import math
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy.stats.sampling import (
from pytest import raises as assert_raises
from scipy import stats
from scipy import special
from scipy.stats import chisquare, cramervonmises
from scipy.stats._distr_params import distdiscrete, distcont
from scipy._lib._util import check_random_state
class TestDiscreteGuideTable:
    basic_fail_dists = {'nchypergeom_fisher', 'nchypergeom_wallenius', 'randint'}

    def test_guide_factor_gt3_raises_warning(self):
        pv = [0.1, 0.3, 0.6]
        urng = np.random.default_rng()
        with pytest.warns(RuntimeWarning):
            DiscreteGuideTable(pv, random_state=urng, guide_factor=7)

    def test_guide_factor_zero_raises_warning(self):
        pv = [0.1, 0.3, 0.6]
        urng = np.random.default_rng()
        with pytest.warns(RuntimeWarning):
            DiscreteGuideTable(pv, random_state=urng, guide_factor=0)

    def test_negative_guide_factor_raises_warning(self):
        pv = [0.1, 0.3, 0.6]
        urng = np.random.default_rng()
        with pytest.warns(RuntimeWarning):
            DiscreteGuideTable(pv, random_state=urng, guide_factor=-1)

    @pytest.mark.parametrize('distname, params', distdiscrete)
    def test_basic(self, distname, params):
        if distname in self.basic_fail_dists:
            msg = 'DGT fails on these probably because of large domains and small computation errors in PMF.'
            pytest.skip(msg)
        if not isinstance(distname, str):
            dist = distname
        else:
            dist = getattr(stats, distname)
        dist = dist(*params)
        domain = dist.support()
        if not np.isfinite(domain[1] - domain[0]):
            pytest.skip('DGT only works with a finite domain.')
        k = np.arange(domain[0], domain[1] + 1)
        pv = dist.pmf(k)
        mv_ex = dist.stats('mv')
        rng = DiscreteGuideTable(dist, random_state=42)
        check_discr_samples(rng, pv, mv_ex)
    u = [np.linspace(0, 1, num=10000), [], [[]], [np.nan], [-np.inf, np.nan, np.inf], 0, [[np.nan, 0.5, 0.1], [0.2, 0.4, np.inf], [-2, 3, 4]]]

    @pytest.mark.parametrize('u', u)
    def test_ppf(self, u):
        n, p = (4, 0.1)
        dist = stats.binom(n, p)
        rng = DiscreteGuideTable(dist, random_state=42)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'invalid value encountered in greater')
            sup.filter(RuntimeWarning, 'invalid value encountered in greater_equal')
            sup.filter(RuntimeWarning, 'invalid value encountered in less')
            sup.filter(RuntimeWarning, 'invalid value encountered in less_equal')
            res = rng.ppf(u)
            expected = stats.binom.ppf(u, n, p)
        assert_equal(res.shape, expected.shape)
        assert_equal(res, expected)

    @pytest.mark.parametrize('pv, msg', bad_pv_common)
    def test_bad_pv(self, pv, msg):
        with pytest.raises(ValueError, match=msg):
            DiscreteGuideTable(pv)
    inf_domain = [(-np.inf, np.inf), (np.inf, np.inf), (-np.inf, -np.inf), (0, np.inf), (-np.inf, 0)]

    @pytest.mark.parametrize('domain', inf_domain)
    def test_inf_domain(self, domain):
        with pytest.raises(ValueError, match='must be finite'):
            DiscreteGuideTable(stats.binom(10, 0.2), domain=domain)