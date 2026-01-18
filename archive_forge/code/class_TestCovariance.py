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
class TestCovariance:

    def test_input_validation(self):
        message = 'The input `precision` must be a square, two-dimensional...'
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaPrecision(np.ones(2))
        message = '`precision.shape` must equal `covariance.shape`.'
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaPrecision(np.eye(3), covariance=np.eye(2))
        message = 'The input `diagonal` must be a one-dimensional array...'
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaDiagonal('alpaca')
        message = 'The input `cholesky` must be a square, two-dimensional...'
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaCholesky(np.ones(2))
        message = 'The input `eigenvalues` must be a one-dimensional...'
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaEigendecomposition(('alpaca', np.eye(2)))
        message = 'The input `eigenvectors` must be a square...'
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaEigendecomposition((np.ones(2), 'alpaca'))
        message = 'The shapes of `eigenvalues` and `eigenvectors` must be...'
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaEigendecomposition(([1, 2, 3], np.eye(2)))
    _covariance_preprocessing = {'Diagonal': np.diag, 'Precision': np.linalg.inv, 'Cholesky': np.linalg.cholesky, 'Eigendecomposition': np.linalg.eigh, 'PSD': lambda x: _PSD(x, allow_singular=True)}
    _all_covariance_types = np.array(list(_covariance_preprocessing))
    _matrices = {'diagonal full rank': np.diag([1, 2, 3]), 'general full rank': [[5, 1, 3], [1, 6, 4], [3, 4, 7]], 'diagonal singular': np.diag([1, 0, 3]), 'general singular': [[5, -1, 0], [-1, 5, 0], [0, 0, 0]]}
    _cov_types = {'diagonal full rank': _all_covariance_types, 'general full rank': _all_covariance_types[1:], 'diagonal singular': _all_covariance_types[[0, -2, -1]], 'general singular': _all_covariance_types[-2:]}

    @pytest.mark.parametrize('cov_type_name', _all_covariance_types[:-1])
    def test_factories(self, cov_type_name):
        A = np.diag([1, 2, 3])
        x = [-4, 2, 5]
        cov_type = getattr(_covariance, f'CovVia{cov_type_name}')
        preprocessing = self._covariance_preprocessing[cov_type_name]
        factory = getattr(Covariance, f'from_{cov_type_name.lower()}')
        res = factory(preprocessing(A))
        ref = cov_type(preprocessing(A))
        assert type(res) == type(ref)
        assert_allclose(res.whiten(x), ref.whiten(x))

    @pytest.mark.parametrize('matrix_type', list(_matrices))
    @pytest.mark.parametrize('cov_type_name', _all_covariance_types)
    def test_covariance(self, matrix_type, cov_type_name):
        message = f'CovVia{cov_type_name} does not support {matrix_type} matrices'
        if cov_type_name not in self._cov_types[matrix_type]:
            pytest.skip(message)
        A = self._matrices[matrix_type]
        cov_type = getattr(_covariance, f'CovVia{cov_type_name}')
        preprocessing = self._covariance_preprocessing[cov_type_name]
        psd = _PSD(A, allow_singular=True)
        cov_object = cov_type(preprocessing(A))
        assert_close(cov_object.log_pdet, psd.log_pdet)
        assert_equal(cov_object.rank, psd.rank)
        assert_equal(cov_object.shape, np.asarray(A).shape)
        assert_close(cov_object.covariance, np.asarray(A))
        rng = np.random.default_rng(5292808890472453840)
        x = rng.random(size=3)
        res = cov_object.whiten(x)
        ref = x @ psd.U
        assert_close(res @ res, ref @ ref)
        if hasattr(cov_object, '_colorize') and 'singular' not in matrix_type:
            assert_close(cov_object.colorize(res), x)
        x = rng.random(size=(2, 4, 3))
        res = cov_object.whiten(x)
        ref = x @ psd.U
        assert_close((res ** 2).sum(axis=-1), (ref ** 2).sum(axis=-1))
        if hasattr(cov_object, '_colorize') and 'singular' not in matrix_type:
            assert_close(cov_object.colorize(res), x)
        if hasattr(cov_object, '_colorize'):
            res = cov_object.colorize(np.eye(len(A)))
            assert_close(res.T @ res, A)

    @pytest.mark.parametrize('size', [None, tuple(), 1, (2, 4, 3)])
    @pytest.mark.parametrize('matrix_type', list(_matrices))
    @pytest.mark.parametrize('cov_type_name', _all_covariance_types)
    def test_mvn_with_covariance(self, size, matrix_type, cov_type_name):
        message = f'CovVia{cov_type_name} does not support {matrix_type} matrices'
        if cov_type_name not in self._cov_types[matrix_type]:
            pytest.skip(message)
        A = self._matrices[matrix_type]
        cov_type = getattr(_covariance, f'CovVia{cov_type_name}')
        preprocessing = self._covariance_preprocessing[cov_type_name]
        mean = [0.1, 0.2, 0.3]
        cov_object = cov_type(preprocessing(A))
        mvn = multivariate_normal
        dist0 = multivariate_normal(mean, A, allow_singular=True)
        dist1 = multivariate_normal(mean, cov_object, allow_singular=True)
        rng = np.random.default_rng(5292808890472453840)
        x = rng.multivariate_normal(mean, A, size=size)
        rng = np.random.default_rng(5292808890472453840)
        x1 = mvn.rvs(mean, cov_object, size=size, random_state=rng)
        rng = np.random.default_rng(5292808890472453840)
        x2 = mvn(mean, cov_object, seed=rng).rvs(size=size)
        if isinstance(cov_object, _covariance.CovViaPSD):
            assert_close(x1, np.squeeze(x))
            assert_close(x2, np.squeeze(x))
        else:
            assert_equal(x1.shape, x.shape)
            assert_equal(x2.shape, x.shape)
            assert_close(x2, x1)
        assert_close(mvn.pdf(x, mean, cov_object), dist0.pdf(x))
        assert_close(dist1.pdf(x), dist0.pdf(x))
        assert_close(mvn.logpdf(x, mean, cov_object), dist0.logpdf(x))
        assert_close(dist1.logpdf(x), dist0.logpdf(x))
        assert_close(mvn.entropy(mean, cov_object), dist0.entropy())
        assert_close(dist1.entropy(), dist0.entropy())

    @pytest.mark.parametrize('size', [tuple(), (2, 4, 3)])
    @pytest.mark.parametrize('cov_type_name', _all_covariance_types)
    def test_mvn_with_covariance_cdf(self, size, cov_type_name):
        matrix_type = 'diagonal full rank'
        A = self._matrices[matrix_type]
        cov_type = getattr(_covariance, f'CovVia{cov_type_name}')
        preprocessing = self._covariance_preprocessing[cov_type_name]
        mean = [0.1, 0.2, 0.3]
        cov_object = cov_type(preprocessing(A))
        mvn = multivariate_normal
        dist0 = multivariate_normal(mean, A, allow_singular=True)
        dist1 = multivariate_normal(mean, cov_object, allow_singular=True)
        rng = np.random.default_rng(5292808890472453840)
        x = rng.multivariate_normal(mean, A, size=size)
        assert_close(mvn.cdf(x, mean, cov_object), dist0.cdf(x))
        assert_close(dist1.cdf(x), dist0.cdf(x))
        assert_close(mvn.logcdf(x, mean, cov_object), dist0.logcdf(x))
        assert_close(dist1.logcdf(x), dist0.logcdf(x))

    def test_covariance_instantiation(self):
        message = 'The `Covariance` class cannot be instantiated directly.'
        with pytest.raises(NotImplementedError, match=message):
            Covariance()

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_gh9942(self):
        A = np.diag([1, 2, -1e-08])
        n = A.shape[0]
        mean = np.zeros(n)
        with pytest.raises(ValueError, match='The input matrix must be...'):
            multivariate_normal(mean, A).rvs()
        seed = 3562050283508273023
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)
        cov = Covariance.from_eigendecomposition(np.linalg.eigh(A))
        rv = multivariate_normal(mean, cov)
        res = rv.rvs(random_state=rng1)
        ref = multivariate_normal.rvs(mean, cov, random_state=rng2)
        assert_equal(res, ref)

    def test_gh19197(self):
        mean = np.ones(2)
        cov = Covariance.from_eigendecomposition((np.zeros(2), np.eye(2)))
        dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
        rvs = dist.rvs(size=None)
        assert_equal(rvs, mean)
        cov = scipy.stats.Covariance.from_eigendecomposition((np.array([1.0, 0.0]), np.array([[1.0, 0.0], [0.0, 400.0]])))
        dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
        rvs = dist.rvs(size=None)
        assert rvs[0] != mean[0]
        assert rvs[1] == mean[1]