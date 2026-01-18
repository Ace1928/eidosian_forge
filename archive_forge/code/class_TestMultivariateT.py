import pickle
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from .test_continuous_basic import check_distribution_rvs
import numpy
import numpy as np
import scipy.linalg
from scipy.stats._multivariate import (_PSD,
from scipy.stats import (multivariate_normal, multivariate_hypergeom,
from scipy.stats import _covariance, Covariance
from scipy import stats
from scipy.integrate import romb, qmc_quad, tplquad
from scipy.special import multigammaln
from scipy._lib._pep440 import Version
from .common_tests import check_random_state_property
from .data._mvt import _qsimvtv
from unittest.mock import patch
class TestMultivariateT:
    PDF_TESTS = [([[1, 2], [4, 1], [2, 1], [2, 4], [1, 4], [4, 1], [3, 2], [3, 3], [4, 4], [5, 1]], [0, 0], [[1, 0], [0, 1]], 4, [0.013972450422333742, 0.001099872190679333, 0.013972450422333742, 0.0007368284402402561, 0.001099872190679333, 0.001099872190679333, 0.0020732579600816823, 0.0009566037150527143, 0.00021831953784896499, 0.0003772561614030115]), ([[0.9718, 0.1298, 0.8134], [0.4922, 0.5522, 0.7185], [0.301, 0.1491, 0.5008], [0.5971, 0.2585, 0.894], [0.5434, 0.5287, 0.9507]], [-1, 1, 50], [[1.0, 0.5, 0.25], [0.5, 1.0, -0.1], [0.25, -0.1, 1.0]], 8, [6.960927969746777e-16, 7.370073905220737e-16, 6.952290996266917e-16, 7.421229355799831e-16, 7.703967515402212e-16])]

    @pytest.mark.parametrize('x, loc, shape, df, ans', PDF_TESTS)
    def test_pdf_correctness(self, x, loc, shape, df, ans):
        dist = multivariate_t(loc, shape, df, seed=0)
        val = dist.pdf(x)
        assert_array_almost_equal(val, ans)

    @pytest.mark.parametrize('x, loc, shape, df, ans', PDF_TESTS)
    def test_logpdf_correct(self, x, loc, shape, df, ans):
        dist = multivariate_t(loc, shape, df, seed=0)
        val1 = dist.pdf(x)
        val2 = dist.logpdf(x)
        assert_array_almost_equal(np.log(val1), val2)

    def test_mvt_with_df_one_is_cauchy(self):
        x = [9, 7, 4, 1, -3, 9, 0, -3, -1, 3]
        val = multivariate_t.pdf(x, df=1)
        ans = cauchy.pdf(x)
        assert_array_almost_equal(val, ans)

    def test_mvt_with_high_df_is_approx_normal(self):
        P_VAL_MIN = 0.1
        dist = multivariate_t(0, 1, df=100000, seed=1)
        samples = dist.rvs(size=100000)
        _, p = normaltest(samples)
        assert p > P_VAL_MIN
        dist = multivariate_t([-2, 3], [[10, -1], [-1, 10]], df=100000, seed=42)
        samples = dist.rvs(size=100000)
        _, p = normaltest(samples)
        assert (p > P_VAL_MIN).all()

    @patch('scipy.stats.multivariate_normal._logpdf')
    def test_mvt_with_inf_df_calls_normal(self, mock):
        dist = multivariate_t(0, 1, df=np.inf, seed=7)
        assert isinstance(dist, multivariate_normal_frozen)
        multivariate_t.pdf(0, df=np.inf)
        assert mock.call_count == 1
        multivariate_t.logpdf(0, df=np.inf)
        assert mock.call_count == 2

    def test_shape_correctness(self):
        dim = 4
        loc = np.zeros(dim)
        shape = np.eye(dim)
        df = 4.5
        x = np.zeros(dim)
        res = multivariate_t(loc, shape, df).pdf(x)
        assert np.isscalar(res)
        res = multivariate_t(loc, shape, df).logpdf(x)
        assert np.isscalar(res)
        n_samples = 7
        x = np.random.random((n_samples, dim))
        res = multivariate_t(loc, shape, df).pdf(x)
        assert res.shape == (n_samples,)
        res = multivariate_t(loc, shape, df).logpdf(x)
        assert res.shape == (n_samples,)
        res = multivariate_t(np.zeros(1), np.eye(1), 1).rvs()
        assert np.isscalar(res)
        size = 7
        res = multivariate_t(np.zeros(1), np.eye(1), 1).rvs(size=size)
        assert res.shape == (size,)

    def test_default_arguments(self):
        dist = multivariate_t()
        assert_equal(dist.loc, [0])
        assert_equal(dist.shape, [[1]])
        assert dist.df == 1
    DEFAULT_ARGS_TESTS = [(None, None, None, 0, 1, 1), (None, None, 7, 0, 1, 7), (None, [[7, 0], [0, 7]], None, [0, 0], [[7, 0], [0, 7]], 1), (None, [[7, 0], [0, 7]], 7, [0, 0], [[7, 0], [0, 7]], 7), ([7, 7], None, None, [7, 7], [[1, 0], [0, 1]], 1), ([7, 7], None, 7, [7, 7], [[1, 0], [0, 1]], 7), ([7, 7], [[7, 0], [0, 7]], None, [7, 7], [[7, 0], [0, 7]], 1), ([7, 7], [[7, 0], [0, 7]], 7, [7, 7], [[7, 0], [0, 7]], 7)]

    @pytest.mark.parametrize('loc, shape, df, loc_ans, shape_ans, df_ans', DEFAULT_ARGS_TESTS)
    def test_default_args(self, loc, shape, df, loc_ans, shape_ans, df_ans):
        dist = multivariate_t(loc=loc, shape=shape, df=df)
        assert_equal(dist.loc, loc_ans)
        assert_equal(dist.shape, shape_ans)
        assert dist.df == df_ans
    ARGS_SHAPES_TESTS = [(-1, 2, 3, [-1], [[2]], 3), ([-1], [2], 3, [-1], [[2]], 3), (np.array([-1]), np.array([2]), 3, [-1], [[2]], 3)]

    @pytest.mark.parametrize('loc, shape, df, loc_ans, shape_ans, df_ans', ARGS_SHAPES_TESTS)
    def test_scalar_list_and_ndarray_arguments(self, loc, shape, df, loc_ans, shape_ans, df_ans):
        dist = multivariate_t(loc, shape, df)
        assert_equal(dist.loc, loc_ans)
        assert_equal(dist.shape, shape_ans)
        assert_equal(dist.df, df_ans)

    def test_argument_error_handling(self):
        loc = [[1, 1]]
        assert_raises(ValueError, multivariate_t, **dict(loc=loc))
        shape = [[1, 1], [2, 2], [3, 3]]
        assert_raises(ValueError, multivariate_t, **dict(loc=loc, shape=shape))
        loc = np.zeros(2)
        shape = np.eye(2)
        df = -1
        assert_raises(ValueError, multivariate_t, **dict(loc=loc, shape=shape, df=df))
        df = 0
        assert_raises(ValueError, multivariate_t, **dict(loc=loc, shape=shape, df=df))

    def test_reproducibility(self):
        rng = np.random.RandomState(4)
        loc = rng.uniform(size=3)
        shape = np.eye(3)
        dist1 = multivariate_t(loc, shape, df=3, seed=2)
        dist2 = multivariate_t(loc, shape, df=3, seed=2)
        samples1 = dist1.rvs(size=10)
        samples2 = dist2.rvs(size=10)
        assert_equal(samples1, samples2)

    def test_allow_singular(self):
        args = dict(loc=[0, 0], shape=[[0, 0], [0, 1]], df=1, allow_singular=False)
        assert_raises(np.linalg.LinAlgError, multivariate_t, **args)

    @pytest.mark.parametrize('size', [(10, 3), (5, 6, 4, 3)])
    @pytest.mark.parametrize('dim', [2, 3, 4, 5])
    @pytest.mark.parametrize('df', [1.0, 2.0, np.inf])
    def test_rvs(self, size, dim, df):
        dist = multivariate_t(np.zeros(dim), np.eye(dim), df)
        rvs = dist.rvs(size=size)
        assert rvs.shape == size + (dim,)

    def test_cdf_signs(self):
        mean = np.zeros(3)
        cov = np.eye(3)
        df = 10
        b = [[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0]]
        a = [[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]]
        expected_signs = np.array([1, -1, -1, 1])
        cdf = multivariate_normal.cdf(b, mean, cov, df, lower_limit=a)
        assert_allclose(cdf, cdf[0] * expected_signs)

    @pytest.mark.parametrize('dim', [1, 2, 5, 10])
    def test_cdf_against_multivariate_normal(self, dim):
        self.cdf_against_mvn_test(dim)

    @pytest.mark.parametrize('dim', [3, 6, 9])
    def test_cdf_against_multivariate_normal_singular(self, dim):
        self.cdf_against_mvn_test(3, True)

    def cdf_against_mvn_test(self, dim, singular=False):
        rng = np.random.default_rng(413722918996573)
        n = 3
        w = 10 ** rng.uniform(-2, 1, size=dim)
        cov = _random_covariance(dim, w, rng, singular)
        mean = 10 ** rng.uniform(-1, 2, size=dim) * np.sign(rng.normal(size=dim))
        a = -10 ** rng.uniform(-1, 2, size=(n, dim)) + mean
        b = 10 ** rng.uniform(-1, 2, size=(n, dim)) + mean
        res = stats.multivariate_t.cdf(b, mean, cov, df=10000, lower_limit=a, allow_singular=True, random_state=rng)
        ref = stats.multivariate_normal.cdf(b, mean, cov, allow_singular=True, lower_limit=a)
        assert_allclose(res, ref, atol=0.0005)

    def test_cdf_against_univariate_t(self):
        rng = np.random.default_rng(413722918996573)
        cov = 2
        mean = 0
        x = rng.normal(size=10, scale=np.sqrt(cov))
        df = 3
        res = stats.multivariate_t.cdf(x, mean, cov, df, lower_limit=-np.inf, random_state=rng)
        ref = stats.t.cdf(x, df, mean, np.sqrt(cov))
        incorrect = stats.norm.cdf(x, mean, np.sqrt(cov))
        assert_allclose(res, ref, atol=0.0005)
        assert np.all(np.abs(res - incorrect) > 0.001)

    @pytest.mark.parametrize('dim', [2, 3, 5, 10])
    @pytest.mark.parametrize('seed', [3363958638, 7891119608, 3887698049, 5013150848, 1495033423, 6170824608])
    @pytest.mark.parametrize('singular', [False, True])
    def test_cdf_against_qsimvtv(self, dim, seed, singular):
        if singular and seed != 3363958638:
            pytest.skip('Agreement with qsimvtv is not great in singular case')
        rng = np.random.default_rng(seed)
        w = 10 ** rng.uniform(-2, 2, size=dim)
        cov = _random_covariance(dim, w, rng, singular)
        mean = rng.random(dim)
        a = -rng.random(dim)
        b = rng.random(dim)
        df = rng.random() * 5
        res = stats.multivariate_t.cdf(b, mean, cov, df, random_state=rng, allow_singular=True)
        with np.errstate(invalid='ignore'):
            ref = _qsimvtv(20000, df, cov, np.inf * a, b - mean, rng)[0]
        assert_allclose(res, ref, atol=0.0002, rtol=0.001)
        res = stats.multivariate_t.cdf(b, mean, cov, df, lower_limit=a, random_state=rng, allow_singular=True)
        with np.errstate(invalid='ignore'):
            ref = _qsimvtv(20000, df, cov, a - mean, b - mean, rng)[0]
        assert_allclose(res, ref, atol=0.0001, rtol=0.001)

    def test_cdf_against_generic_integrators(self):
        dim = 3
        rng = np.random.default_rng(41372291899657)
        w = 10 ** rng.uniform(-1, 1, size=dim)
        cov = _random_covariance(dim, w, rng, singular=True)
        mean = rng.random(dim)
        a = -rng.random(dim)
        b = rng.random(dim)
        df = rng.random() * 5
        res = stats.multivariate_t.cdf(b, mean, cov, df, random_state=rng, lower_limit=a)

        def integrand(x):
            return stats.multivariate_t.pdf(x.T, mean, cov, df)
        ref = qmc_quad(integrand, a, b, qrng=stats.qmc.Halton(d=dim, seed=rng))
        assert_allclose(res, ref.integral, rtol=0.001)

        def integrand(*zyx):
            return stats.multivariate_t.pdf(zyx[::-1], mean, cov, df)
        ref = tplquad(integrand, a[0], b[0], a[1], b[1], a[2], b[2])
        assert_allclose(res, ref[0], rtol=0.001)

    def test_against_matlab(self):
        rng = np.random.default_rng(2967390923)
        cov = np.array([[6.21786909, 0.2333667, 7.95506077], [0.2333667, 29.67390923, 16.53946426], [7.95506077, 16.53946426, 19.17725252]])
        df = 1.9559939787727658
        dist = stats.multivariate_t(shape=cov, df=df)
        res = dist.cdf([0, 0, 0], random_state=rng)
        ref = 0.2523
        assert_allclose(res, ref, rtol=0.001)

    def test_frozen(self):
        seed = 4137229573
        rng = np.random.default_rng(seed)
        loc = rng.uniform(size=3)
        x = rng.uniform(size=3) + loc
        shape = np.eye(3)
        df = rng.random()
        args = (loc, shape, df)
        rng_frozen = np.random.default_rng(seed)
        rng_unfrozen = np.random.default_rng(seed)
        dist = stats.multivariate_t(*args, seed=rng_frozen)
        assert_equal(dist.cdf(x), multivariate_t.cdf(x, *args, random_state=rng_unfrozen))

    def test_vectorized(self):
        dim = 4
        n = (2, 3)
        rng = np.random.default_rng(413722918996573)
        A = rng.random(size=(dim, dim))
        cov = A @ A.T
        mean = rng.random(dim)
        x = rng.random(n + (dim,))
        df = rng.random() * 5
        res = stats.multivariate_t.cdf(x, mean, cov, df, random_state=rng)

        def _cdf_1d(x):
            return _qsimvtv(10000, df, cov, -np.inf * x, x - mean, rng)[0]
        ref = np.apply_along_axis(_cdf_1d, -1, x)
        assert_allclose(res, ref, atol=0.0001, rtol=0.001)

    @pytest.mark.parametrize('dim', (3, 7))
    def test_against_analytical(self, dim):
        rng = np.random.default_rng(413722918996573)
        A = scipy.linalg.toeplitz(c=[1] + [0.5] * (dim - 1))
        res = stats.multivariate_t(shape=A).cdf([0] * dim, random_state=rng)
        ref = 1 / (dim + 1)
        assert_allclose(res, ref, rtol=5e-05)

    def test_entropy_inf_df(self):
        cov = np.eye(3, 3)
        df = np.inf
        mvt_entropy = stats.multivariate_t.entropy(shape=cov, df=df)
        mvn_entropy = stats.multivariate_normal.entropy(None, cov)
        assert mvt_entropy == mvn_entropy

    @pytest.mark.parametrize('df', [1, 10, 100])
    def test_entropy_1d(self, df):
        mvt_entropy = stats.multivariate_t.entropy(shape=1.0, df=df)
        t_entropy = stats.t.entropy(df=df)
        assert_allclose(mvt_entropy, t_entropy, rtol=1e-13)

    @pytest.mark.parametrize('df, cov, ref, tol', [(10, np.eye(2, 2), 3.0378770664093313, 1e-14), (100, np.array([[0.5, 1], [1, 10]]), 3.55102424550609, 1e-08)])
    def test_entropy_vs_numerical_integration(self, df, cov, ref, tol):
        loc = np.zeros((2,))
        mvt = stats.multivariate_t(loc, cov, df)
        assert_allclose(mvt.entropy(), ref, rtol=tol)

    @pytest.mark.parametrize('df, dim, ref, tol', [(10, 1, 1.5212624929756808, 1e-15), (100, 1, 1.4289633653182439, 1e-13), (500, 1, 1.420939531869349, 1e-14), (1e+20, 1, 1.4189385332046727, 1e-15), (1e+100, 1, 1.4189385332046727, 1e-15), (10, 10, 15.069150450832911, 1e-15), (1000, 10, 14.19936546446673, 1e-13), (1e+20, 10, 14.189385332046728, 1e-15), (1e+100, 10, 14.189385332046728, 1e-15), (10, 100, 148.28902883192654, 1e-15), (1000, 100, 141.99155538003762, 1e-14), (1e+20, 100, 141.8938533204673, 1e-15), (1e+100, 100, 141.8938533204673, 1e-15)])
    def test_extreme_entropy(self, df, dim, ref, tol):
        mvt = stats.multivariate_t(shape=np.eye(dim), df=df)
        assert_allclose(mvt.entropy(), ref, rtol=tol)

    def test_entropy_with_covariance(self):
        _A = np.array([[1.42, 0.09, -0.49, 0.17, 0.74], [-1.13, -0.01, 0.71, 0.4, -0.56], [1.07, 0.44, -0.28, -0.44, 0.29], [-1.5, -0.94, -0.67, 0.73, -1.1], [0.17, -0.08, 1.46, -0.32, 1.36]])
        cov = _A @ _A.T
        df = 1e+20
        mul_t_entropy = stats.multivariate_t.entropy(shape=cov, df=df)
        mul_norm_entropy = multivariate_normal(None, cov=cov).entropy()
        assert_allclose(mul_t_entropy, mul_norm_entropy, rtol=1e-15)
        df1 = 765
        df2 = 768
        _entropy1 = stats.multivariate_t.entropy(shape=cov, df=df1)
        _entropy2 = stats.multivariate_t.entropy(shape=cov, df=df2)
        assert_allclose(_entropy1, _entropy2, rtol=1e-05)