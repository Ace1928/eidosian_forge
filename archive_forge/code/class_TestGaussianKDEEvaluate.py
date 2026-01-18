from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
class TestGaussianKDEEvaluate:

    def test_evaluate_diff_dim(self):
        """
        Test the evaluate method when the dim's of dataset and points have
        different dimensions.
        """
        x1 = np.arange(3, 10, 2)
        kde = mlab.GaussianKDE(x1)
        x2 = np.arange(3, 12, 2)
        y_expected = [0.08797252, 0.11774109, 0.11774109, 0.08797252, 0.0370153]
        y = kde.evaluate(x2)
        np.testing.assert_array_almost_equal(y, y_expected, 7)

    def test_evaluate_inv_dim(self):
        """
        Invert the dimensions; i.e., for a dataset of dimension 1 [3, 2, 4],
        the points should have a dimension of 3 [[3], [2], [4]].
        """
        np.random.seed(8765678)
        n_basesample = 50
        multidim_data = np.random.randn(n_basesample)
        kde = mlab.GaussianKDE(multidim_data)
        x2 = [[1], [2], [3]]
        with pytest.raises(ValueError):
            kde.evaluate(x2)

    def test_evaluate_dim_and_num(self):
        """Tests if evaluated against a one by one array"""
        x1 = np.arange(3, 10, 2)
        x2 = np.array([3])
        kde = mlab.GaussianKDE(x1)
        y_expected = [0.08797252]
        y = kde.evaluate(x2)
        np.testing.assert_array_almost_equal(y, y_expected, 7)

    def test_evaluate_point_dim_not_one(self):
        x1 = np.arange(3, 10, 2)
        x2 = [np.arange(3, 10, 2), np.arange(3, 10, 2)]
        kde = mlab.GaussianKDE(x1)
        with pytest.raises(ValueError):
            kde.evaluate(x2)

    def test_evaluate_equal_dim_and_num_lt(self):
        x1 = np.arange(3, 10, 2)
        x2 = np.arange(3, 8, 2)
        kde = mlab.GaussianKDE(x1)
        y_expected = [0.08797252, 0.11774109, 0.11774109]
        y = kde.evaluate(x2)
        np.testing.assert_array_almost_equal(y, y_expected, 7)