import numpy as np
from scipy import stats
from statsmodels.sandbox.nonparametric import kernels
from statsmodels.distributions.mixture_rvs import mixture_rvs
from statsmodels.nonparametric.bandwidths import select_bandwidth
from statsmodels.nonparametric.bandwidths import bw_normal_reference
from numpy.testing import assert_allclose
import pytest
class TestAnyBandwidthZero(BandwidthZero):
    xx = np.random.normal(size=(100, 3))
    xx[:, 0] = 1.0