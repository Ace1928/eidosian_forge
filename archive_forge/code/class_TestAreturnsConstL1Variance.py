import os
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.tsa.regime_switching import (markov_switching,
class TestAreturnsConstL1Variance(MarkovRegression):

    @classmethod
    def setup_class(cls):
        true = {'params': np.r_[0.7530865, 0.6825357, 0.7641424, 1.972771, 0.0790744, 0.527953, 0.5895792 ** 2, 1.605333 ** 2], 'llf': -745.7977, 'llf_fit': -745.7977, 'llf_fit_em': -745.83654, 'bse_oim': np.r_[0.0634387, 0.0662574, 0.0782852, 0.2784204, 0.0301862, 0.0857841, np.nan, np.nan]}
        super().setup_class(true, areturns[1:], k_regimes=2, exog=areturns[:-1], switching_variance=True)

    def test_fit(self, **kwargs):
        kwargs.setdefault('em_iter', 10)
        kwargs.setdefault('maxiter', 100)
        super().test_fit(**kwargs)

    def test_bse(self):
        bse = self.result.cov_params_approx.diagonal() ** 0.5
        assert_allclose(bse[:-2], self.true['bse_oim'][:-2], atol=1e-07)