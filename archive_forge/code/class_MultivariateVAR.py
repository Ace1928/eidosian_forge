import os
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, structural, varmax
from statsmodels.tsa.statespace.simulation_smoother import (
class MultivariateVAR:
    """
    More generic tests for simulation smoothing; use actual N(0,1) variates
    """

    @classmethod
    def setup_class(cls, missing='none', *args, **kwargs):
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start='1959-01-01', end='2009-7-01', freq='QS')
        obs = np.log(dta[['realgdp', 'realcons', 'realinv']]).diff().iloc[1:]
        if missing == 'all':
            obs.iloc[0:50, :] = np.nan
        elif missing == 'partial':
            obs.iloc[0:50, 0] = np.nan
        elif missing == 'mixed':
            obs.iloc[0:50, 0] = np.nan
            obs.iloc[19:70, 1] = np.nan
            obs.iloc[39:90, 2] = np.nan
            obs.iloc[119:130, 0] = np.nan
            obs.iloc[119:130, 2] = np.nan
            obs.iloc[-10:, :] = np.nan
        mod = mlemodel.MLEModel(obs, k_states=3, k_posdef=3, **kwargs)
        mod['design'] = np.eye(3)
        mod['obs_cov'] = np.array([[6.40649e-05, 0.0, 0.0], [0.0, 5.72802e-05, 0.0], [0.0, 0.0, 0.0017088585]])
        mod['transition'] = np.array([[-0.1119908792, 0.8441841604, 0.0238725303], [0.2629347724, 0.4996718412, -0.0173023305], [-3.2192369082, 4.1536028244, 0.4514379215]])
        mod['selection'] = np.eye(3)
        mod['state_cov'] = np.array([[6.40649e-05, 3.88496e-05, 0.0002148769], [3.88496e-05, 5.72802e-05, 1.555e-06], [0.0002148769, 1.555e-06, 0.0017088585]])
        mod.initialize_approximate_diffuse(1000000.0)
        mod.ssm.filter_univariate = True
        cls.model = mod
        cls.results = mod.smooth([], return_ssm=True)
        cls.sim = cls.model.simulation_smoother()

    def test_loglike(self):
        assert_allclose(np.sum(self.results.llf_obs), self.true_llf)

    def test_simulation_smoothing(self):
        sim = self.sim
        Z = self.model['design']
        nobs = self.model.nobs
        k_endog = self.model.k_endog
        sim.simulate(measurement_disturbance_variates=self.variates[:nobs * k_endog], state_disturbance_variates=self.variates[nobs * k_endog:-3], initial_state_variates=self.variates[-3:])
        assert_allclose(sim.simulated_state.T, self.true[['state1', 'state2', 'state3']], atol=1e-07)
        assert_allclose(sim.simulated_measurement_disturbance, self.true[['eps1', 'eps2', 'eps3']].T, atol=1e-07)
        assert_allclose(sim.simulated_state_disturbance, self.true[['eta1', 'eta2', 'eta3']].T, atol=1e-07)
        signals = np.zeros((3, self.model.nobs))
        for t in range(self.model.nobs):
            signals[:, t] = np.dot(Z, sim.simulated_state[:, t])
        assert_allclose(signals, self.true[['signal1', 'signal2', 'signal3']].T, atol=1e-07)