import numpy as np
from numpy.testing import assert_allclose, assert_array_less
from scipy import stats
import pytest
import statsmodels.nonparametric.kernels_asymmetric as kern
class TestKernelsUnit(CheckKernels):

    @classmethod
    def setup_class(cls):
        np.random.seed(987456)
        nobs = 1000
        distr0 = stats.beta(2, 3)
        rvs = distr0.rvs(size=nobs)
        x_plot = np.linspace(1e-10, 1, 51)
        cls.rvs = rvs
        cls.x_plot = x_plot
        cls.pdf_dgp = distr0.pdf(x_plot)
        cls.cdf_dgp = distr0.cdf(x_plot)
        cls.amse_pdf = 0.01
        cls.amse_cdf = 0.005

    @pytest.mark.parametrize('case', kernels_unit)
    def test_kernels(self, case):
        super().test_kernels(case)

    @pytest.mark.parametrize('case', kernels_unit)
    def test_kernels_vectorized(self, case):
        super().test_kernels_vectorized(case)

    @pytest.mark.parametrize('case', kernels_unit)
    def test_kernels_weights(self, case):
        super().test_kernels_weights(case)