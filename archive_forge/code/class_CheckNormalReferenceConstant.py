import numpy as np
from scipy import stats
from statsmodels.sandbox.nonparametric import kernels
from statsmodels.distributions.mixture_rvs import mixture_rvs
from statsmodels.nonparametric.bandwidths import select_bandwidth
from statsmodels.nonparametric.bandwidths import bw_normal_reference
from numpy.testing import assert_allclose
import pytest
class CheckNormalReferenceConstant:

    def test_calculate_normal_reference_constant(self):
        const = self.constant
        kern = self.kern
        assert_allclose(const, kern.normal_reference_constant, 0.01)