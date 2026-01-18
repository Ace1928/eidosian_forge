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
class TestNumericalInversePolynomial:

    class dist0:

        def pdf(self, x):
            return 3 / 4 * (1 - x * x)

        def cdf(self, x):
            return 3 / 4 * (x - x ** 3 / 3 + 2 / 3)

        def support(self):
            return (-1, 1)

    class dist1:

        def pdf(self, x):
            return stats.norm._pdf(x / 0.1)

        def cdf(self, x):
            return stats.norm._cdf(x / 0.1)

    class dist2:

        def pdf(self, x):
            return 0.05 + 0.45 * (1 + np.sin(2 * np.pi * x))

        def cdf(self, x):
            return 0.05 * (x + 1) + 0.9 * (1.0 + 2.0 * np.pi * (1 + x) - np.cos(2.0 * np.pi * x)) / (4.0 * np.pi)

        def support(self):
            return (-1, 1)

    class dist3:

        def pdf(self, x):
            return 0.2 * (0.05 + 0.45 * (1 + np.sin(2 * np.pi * x)))

        def cdf(self, x):
            return x / 10.0 + 0.5 + 0.09 / (2 * np.pi) * (np.cos(10 * np.pi) - np.cos(2 * np.pi * x))

        def support(self):
            return (-5, 5)
    dists = [dist0(), dist1(), dist2(), dist3()]
    mv0 = [0.0, 4.0 / 15.0]
    mv1 = [0.0, 0.01]
    mv2 = [-0.45 / np.pi, 2 / 3 * 0.5 - 0.45 ** 2 / np.pi ** 2]
    mv3 = [-0.45 / np.pi, 0.2 * 250 / 3 * 0.5 - 0.45 ** 2 / np.pi ** 2]
    mvs = [mv0, mv1, mv2, mv3]

    @pytest.mark.parametrize('dist, mv_ex', zip(dists, mvs))
    def test_basic(self, dist, mv_ex):
        rng = NumericalInversePolynomial(dist, random_state=42)
        check_cont_samples(rng, dist, mv_ex)

    @pytest.mark.xslow
    @pytest.mark.parametrize('distname, params', distcont)
    def test_basic_all_scipy_dists(self, distname, params):
        very_slow_dists = ['anglit', 'gausshyper', 'kappa4', 'ksone', 'kstwo', 'levy_l', 'levy_stable', 'studentized_range', 'trapezoid', 'triang', 'vonmises']
        fail_dists = ['chi2', 'fatiguelife', 'gibrat', 'halfgennorm', 'lognorm', 'ncf', 'ncx2', 'pareto', 't']
        skip_sample_moment_check = ['rel_breitwigner']
        if distname in very_slow_dists:
            pytest.skip(f'PINV too slow for {distname}')
        if distname in fail_dists:
            pytest.skip(f'PINV fails for {distname}')
        dist = getattr(stats, distname) if isinstance(distname, str) else distname
        dist = dist(*params)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            rng = NumericalInversePolynomial(dist, random_state=42)
        if distname in skip_sample_moment_check:
            return
        check_cont_samples(rng, dist, [dist.mean(), dist.var()])

    @pytest.mark.parametrize('pdf, err, msg', bad_pdfs_common)
    def test_bad_pdf(self, pdf, err, msg):

        class dist:
            pass
        dist.pdf = pdf
        with pytest.raises(err, match=msg):
            NumericalInversePolynomial(dist, domain=[0, 5])

    @pytest.mark.parametrize('logpdf, err, msg', bad_logpdfs_common)
    def test_bad_logpdf(self, logpdf, err, msg):

        class dist:
            pass
        dist.logpdf = logpdf
        with pytest.raises(err, match=msg):
            NumericalInversePolynomial(dist, domain=[0, 5])

    @pytest.mark.parametrize('domain, err, msg', inf_nan_domains)
    def test_inf_nan_domains(self, domain, err, msg):
        with pytest.raises(err, match=msg):
            NumericalInversePolynomial(StandardNormal(), domain=domain)
    u = [np.linspace(0, 1, num=10000), [], [[]], [np.nan], [-np.inf, np.nan, np.inf], 0, [[np.nan, 0.5, 0.1], [0.2, 0.4, np.inf], [-2, 3, 4]]]

    @pytest.mark.parametrize('u', u)
    def test_ppf(self, u):
        dist = StandardNormal()
        rng = NumericalInversePolynomial(dist, u_resolution=1e-14)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'invalid value encountered in greater')
            sup.filter(RuntimeWarning, 'invalid value encountered in greater_equal')
            sup.filter(RuntimeWarning, 'invalid value encountered in less')
            sup.filter(RuntimeWarning, 'invalid value encountered in less_equal')
            res = rng.ppf(u)
            expected = stats.norm.ppf(u)
        assert_allclose(res, expected, rtol=1e-11, atol=1e-11)
        assert res.shape == expected.shape
    x = [np.linspace(-10, 10, num=10000), [], [[]], [np.nan], [-np.inf, np.nan, np.inf], 0, [[np.nan, 0.5, 0.1], [0.2, 0.4, np.inf], [-np.inf, 3, 4]]]

    @pytest.mark.parametrize('x', x)
    def test_cdf(self, x):
        dist = StandardNormal()
        rng = NumericalInversePolynomial(dist, u_resolution=1e-14)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'invalid value encountered in greater')
            sup.filter(RuntimeWarning, 'invalid value encountered in greater_equal')
            sup.filter(RuntimeWarning, 'invalid value encountered in less')
            sup.filter(RuntimeWarning, 'invalid value encountered in less_equal')
            res = rng.cdf(x)
            expected = stats.norm.cdf(x)
        assert_allclose(res, expected, rtol=1e-11, atol=1e-11)
        assert res.shape == expected.shape

    def test_u_error(self):
        dist = StandardNormal()
        rng = NumericalInversePolynomial(dist, u_resolution=1e-10)
        max_error, mae = rng.u_error()
        assert max_error < 1e-10
        assert mae <= max_error
        rng = NumericalInversePolynomial(dist, u_resolution=1e-14)
        max_error, mae = rng.u_error()
        assert max_error < 1e-14
        assert mae <= max_error
    bad_orders = [1, 4.5, 20, np.inf, np.nan]
    bad_u_resolution = [1e-20, 0.1, np.inf, np.nan]

    @pytest.mark.parametrize('order', bad_orders)
    def test_bad_orders(self, order):
        dist = StandardNormal()
        msg = '`order` must be an integer in the range \\[3, 17\\].'
        with pytest.raises(ValueError, match=msg):
            NumericalInversePolynomial(dist, order=order)

    @pytest.mark.parametrize('u_resolution', bad_u_resolution)
    def test_bad_u_resolution(self, u_resolution):
        msg = '`u_resolution` must be between 1e-15 and 1e-5.'
        with pytest.raises(ValueError, match=msg):
            NumericalInversePolynomial(StandardNormal(), u_resolution=u_resolution)

    def test_bad_args(self):

        class BadDist:

            def cdf(self, x):
                return stats.norm._cdf(x)
        dist = BadDist()
        msg = 'Either of the methods `pdf` or `logpdf` must be specified'
        with pytest.raises(ValueError, match=msg):
            rng = NumericalInversePolynomial(dist)
        dist = StandardNormal()
        rng = NumericalInversePolynomial(dist)
        msg = '`sample_size` must be greater than or equal to 1000.'
        with pytest.raises(ValueError, match=msg):
            rng.u_error(10)

        class Distribution:

            def pdf(self, x):
                return np.exp(-0.5 * x * x)
        dist = Distribution()
        rng = NumericalInversePolynomial(dist)
        msg = 'Exact CDF required but not found.'
        with pytest.raises(ValueError, match=msg):
            rng.u_error()

    def test_logpdf_pdf_consistency(self):

        class MyDist:
            pass
        dist_pdf = MyDist()
        dist_pdf.pdf = lambda x: math.exp(-x * x / 2)
        rng1 = NumericalInversePolynomial(dist_pdf)
        dist_logpdf = MyDist()
        dist_logpdf.logpdf = lambda x: -x * x / 2
        rng2 = NumericalInversePolynomial(dist_logpdf)
        q = np.linspace(1e-05, 1 - 1e-05, num=100)
        assert_allclose(rng1.ppf(q), rng2.ppf(q))