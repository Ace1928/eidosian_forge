import os
import numpy.testing as npt
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.distributions.mixture_rvs import mixture_rvs
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
import statsmodels.sandbox.nonparametric.kernels as kernels
import statsmodels.nonparametric.bandwidths as bandwidths
class TestKDEBiweight(CheckKDE):

    @classmethod
    def setup_class(cls):
        res1 = KDE(Xi)
        res1.fit(kernel='biw', fft=False, bw='silverman')
        cls.res1 = res1
        cls.res_density = KDEResults['biw_d']