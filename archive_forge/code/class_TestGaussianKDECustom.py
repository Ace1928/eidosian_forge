from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
class TestGaussianKDECustom:

    def test_no_data(self):
        """Pass no data into the GaussianKDE class."""
        with pytest.raises(ValueError):
            mlab.GaussianKDE([])

    def test_single_dataset_element(self):
        """Pass a single dataset element into the GaussianKDE class."""
        with pytest.raises(ValueError):
            mlab.GaussianKDE([42])

    def test_silverman_multidim_dataset(self):
        """Test silverman's for a multi-dimensional array."""
        x1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(np.linalg.LinAlgError):
            mlab.GaussianKDE(x1, 'silverman')

    def test_silverman_singledim_dataset(self):
        """Test silverman's output for a single dimension list."""
        x1 = np.array([-7, -5, 1, 4, 5])
        mygauss = mlab.GaussianKDE(x1, 'silverman')
        y_expected = 0.767703899274755
        assert_almost_equal(mygauss.covariance_factor(), y_expected, 7)

    def test_scott_multidim_dataset(self):
        """Test scott's output for a multi-dimensional array."""
        x1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(np.linalg.LinAlgError):
            mlab.GaussianKDE(x1, 'scott')

    def test_scott_singledim_dataset(self):
        """Test scott's output a single-dimensional array."""
        x1 = np.array([-7, -5, 1, 4, 5])
        mygauss = mlab.GaussianKDE(x1, 'scott')
        y_expected = 0.7247796636776955
        assert_almost_equal(mygauss.covariance_factor(), y_expected, 7)

    def test_scalar_empty_dataset(self):
        """Test the scalar's cov factor for an empty array."""
        with pytest.raises(ValueError):
            mlab.GaussianKDE([], bw_method=5)

    def test_scalar_covariance_dataset(self):
        """Test a scalar's cov factor."""
        np.random.seed(8765678)
        n_basesample = 50
        multidim_data = [np.random.randn(n_basesample) for i in range(5)]
        kde = mlab.GaussianKDE(multidim_data, bw_method=0.5)
        assert kde.covariance_factor() == 0.5

    def test_callable_covariance_dataset(self):
        """Test the callable's cov factor for a multi-dimensional array."""
        np.random.seed(8765678)
        n_basesample = 50
        multidim_data = [np.random.randn(n_basesample) for i in range(5)]

        def callable_fun(x):
            return 0.55
        kde = mlab.GaussianKDE(multidim_data, bw_method=callable_fun)
        assert kde.covariance_factor() == 0.55

    def test_callable_singledim_dataset(self):
        """Test the callable's cov factor for a single-dimensional array."""
        np.random.seed(8765678)
        n_basesample = 50
        multidim_data = np.random.randn(n_basesample)
        kde = mlab.GaussianKDE(multidim_data, bw_method='silverman')
        y_expected = 0.4843884136334891
        assert_almost_equal(kde.covariance_factor(), y_expected, 7)

    def test_wrong_bw_method(self):
        """Test the error message that should be called when bw is invalid."""
        np.random.seed(8765678)
        n_basesample = 50
        data = np.random.randn(n_basesample)
        with pytest.raises(ValueError):
            mlab.GaussianKDE(data, bw_method='invalid')