import os
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.tsa.regime_switching import (markov_switching,
class TestFedFundsConstL1Exog(MarkovRegression):

    @classmethod
    def setup_class(cls):
        path = os.path.join(current_path, 'results', 'results_predict_fedfunds.csv')
        results = pd.read_csv(path)
        true = {'params': np.r_[0.7279288, 0.2114578, 0.6554954, -0.0944924, 0.8314458, 0.9292574, 0.1355425, 0.0343072, -0.0273928, 0.2125275, 0.5764495 ** 2], 'llf': -229.25614, 'llf_fit': -229.25614, 'llf_fit_em': -229.25624, 'bse_oim': np.r_[0.0929915, 0.0641179, 0.1373889, 0.1279231, 0.0333236, 0.0270852, 0.0294113, 0.0240138, 0.0408057, 0.0297351, np.nan], 'predict0': results.iloc[4:]['constL1exog_syhat1'], 'predict1': results.iloc[4:]['constL1exog_syhat2'], 'predict_smoothed': results.iloc[4:]['constL1exog_syhat']}
        super().setup_class(true, fedfunds[4:], k_regimes=2, exog=np.c_[fedfunds[3:-1], ogap[4:], inf[4:]])

    def test_fit(self, **kwargs):
        kwargs.setdefault('em_iter', 10)
        kwargs.setdefault('maxiter', 100)
        super().test_fit(**kwargs)

    def test_predict(self):
        for name in ['predicted', 'filtered', 'smoothed', None]:
            actual = self.model.predict(self.true['params'], probabilities=name, conditional=True)
            assert_allclose(actual[0], self.true['predict0'], atol=1e-05)
            assert_allclose(actual[1], self.true['predict1'], atol=1e-05)
        actual = self.model.predict(self.true['params'], probabilities='smoothed')
        assert_allclose(actual, self.true['predict_smoothed'], atol=1e-05)
        actual = self.model.predict(self.true['params'], probabilities=None)
        assert_allclose(actual, self.true['predict_smoothed'], atol=1e-05)
        actual = self.result.predict(probabilities='smoothed')
        assert_allclose(actual, self.true['predict_smoothed'], atol=1e-05)
        actual = self.result.predict(probabilities=None)
        assert_allclose(actual, self.true['predict_smoothed'], atol=1e-05)

    def test_bse(self):
        bse = self.result.cov_params_approx.diagonal() ** 0.5
        assert_allclose(bse[:-1], self.true['bse_oim'][:-1], atol=1e-07)