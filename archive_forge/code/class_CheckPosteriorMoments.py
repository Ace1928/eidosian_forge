import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from scipy.linalg import cho_solve_banded
from statsmodels import datasets
from statsmodels.tsa.statespace import (sarimax, structural, dynamic_factor,
class CheckPosteriorMoments:

    @classmethod
    def setup_class(cls, model_class, missing=None, mean_atol=0, cov_atol=0, use_complex=False, *args, **kwargs):
        cls.mean_atol = mean_atol
        cls.cov_atol = cov_atol
        endog = dta.copy()
        if missing == 'all':
            endog.iloc[0:50, :] = np.nan
        elif missing == 'partial':
            endog.iloc[0:50, 0] = np.nan
        elif missing == 'mixed':
            endog.iloc[0:50, 0] = np.nan
            endog.iloc[19:70, 1] = np.nan
            endog.iloc[39:90, 2] = np.nan
            endog.iloc[119:130, 0] = np.nan
            endog.iloc[119:130, 2] = np.nan
            endog.iloc[-10:, :] = np.nan
        if model_class in [sarimax.SARIMAX, structural.UnobservedComponents]:
            endog = endog.iloc[:, 2]
        cls.mod = model_class(endog, *args, **kwargs)
        params = cls.mod.start_params
        if use_complex:
            params = params + 0j
        cls.res = cls.mod.smooth(params)
        cls.sim_cfa = cls.mod.simulation_smoother(method='cfa')
        cls.sim_cfa.simulate()
        prefix = 'z' if use_complex else 'd'
        cls._sim_cfa = cls.sim_cfa._simulation_smoothers[prefix]

    def test_posterior_mean(self):
        actual = np.array(self._sim_cfa.posterior_mean, copy=True)
        assert_allclose(actual, self.res.smoothed_state, atol=self.mean_atol)
        assert_allclose(self.sim_cfa.posterior_mean, self.res.smoothed_state, atol=self.mean_atol)

    def test_posterior_cov(self):
        inv_chol = np.array(self._sim_cfa.posterior_cov_inv_chol, copy=True)
        actual = cho_solve_banded((inv_chol, True), np.eye(inv_chol.shape[1]))
        for t in range(self.mod.nobs):
            tm = t * self.mod.k_states
            t1m = tm + self.mod.k_states
            assert_allclose(actual[tm:t1m, tm:t1m], self.res.smoothed_state_cov[..., t], atol=self.cov_atol)
        actual = self.sim_cfa.posterior_cov
        for t in range(self.mod.nobs):
            tm = t * self.mod.k_states
            t1m = tm + self.mod.k_states
            assert_allclose(actual[tm:t1m, tm:t1m], self.res.smoothed_state_cov[..., t], atol=self.cov_atol)