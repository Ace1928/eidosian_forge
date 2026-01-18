import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_raises)
from statsmodels.base.transform import (BoxCox)
from statsmodels.datasets import macrodata
class TestTransform:

    @classmethod
    def setup_class(cls):
        data = macrodata.load_pandas()
        cls.x = data.data['realgdp'].values
        cls.bc = BoxCox()

    def test_nonpositive(self):
        y = [1, -1, 1]
        assert_raises(ValueError, self.bc.transform_boxcox, y)
        y = [1, 0, 1]
        assert_raises(ValueError, self.bc.transform_boxcox, y)

    def test_invalid_bounds(self):
        assert_raises(ValueError, self.bc._est_lambda, self.x, (-3, 2, 3))
        assert_raises(ValueError, self.bc._est_lambda, self.x, (2, -1))

    def test_unclear_methods(self):
        assert_raises(ValueError, self.bc._est_lambda, self.x, (-1, 2), 'test')
        assert_raises(ValueError, self.bc.untransform_boxcox, self.x, 1, 'test')

    def test_unclear_scale_parameter(self):
        assert_raises(ValueError, self.bc._est_lambda, self.x, scale='test')
        self.bc._est_lambda(self.x, scale='mad')
        self.bc._est_lambda(self.x, scale='MAD')
        self.bc._est_lambda(self.x, scale='sd')
        self.bc._est_lambda(self.x, scale='SD')

    def test_valid_guerrero(self):
        lmbda = self.bc._est_lambda(self.x, method='guerrero', window_length=4)
        assert_almost_equal(lmbda, 0.507624, 4)
        lmbda = self.bc._est_lambda(self.x, method='guerrero', window_length=2)
        assert_almost_equal(lmbda, 0.513893, 4)

    def test_guerrero_robust_scale(self):
        lmbda = self.bc._est_lambda(self.x, scale='mad')
        assert_almost_equal(lmbda, 0.488621, 4)

    def test_loglik_lambda_estimation(self):
        lmbda = self.bc._est_lambda(self.x, method='loglik')
        assert_almost_equal(lmbda, 0.2, 1)

    def test_boxcox_transformation_methods(self):
        y_transformed_no_lambda = self.bc.transform_boxcox(self.x)
        y_transformed_lambda = self.bc.transform_boxcox(self.x, 0.507624)
        assert_almost_equal(y_transformed_no_lambda[0], y_transformed_lambda[0], 3)
        y, lmbda = self.bc.transform_boxcox(np.arange(1, 100))
        assert_almost_equal(lmbda, 1.0, 5)

    def test_zero_lambda(self):
        y_transform_zero_lambda, lmbda = self.bc.transform_boxcox(self.x, 0.0)
        assert_equal(lmbda, 0.0)
        assert_almost_equal(y_transform_zero_lambda, np.log(self.x), 5)

    def test_naive_back_transformation(self):
        y_zero_lambda = self.bc.transform_boxcox(self.x, 0.0)
        y_half_lambda = self.bc.transform_boxcox(self.x, 0.5)
        y_zero_lambda_un = self.bc.untransform_boxcox(*y_zero_lambda, method='naive')
        y_half_lambda_un = self.bc.untransform_boxcox(*y_half_lambda, method='naive')
        assert_almost_equal(self.x, y_zero_lambda_un, 5)
        assert_almost_equal(self.x, y_half_lambda_un, 5)