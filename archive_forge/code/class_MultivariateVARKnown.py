import os
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, structural, varmax
from statsmodels.tsa.statespace.simulation_smoother import (
class MultivariateVARKnown:
    """
    Tests for simulation smoothing values in a couple of special cases of
    variates. Both computed values and KFAS values are used for comparison
    against the simulation smoother output.
    """

    @classmethod
    def setup_class(cls, missing=None, test_against_KFAS=True, *args, **kwargs):
        cls.test_against_KFAS = test_against_KFAS
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
        if test_against_KFAS:
            obs = obs.iloc[:9]
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

    def test_simulate_0(self):
        n = 10
        measurement_shocks = np.zeros((n, self.model.k_endog))
        state_shocks = np.zeros((n, self.model.ssm.k_posdef))
        initial_state = np.zeros(self.model.k_states)
        obs, states = self.model.ssm.simulate(nsimulations=n, measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
        assert_allclose(obs, np.zeros((n, self.model.k_endog)))
        assert_allclose(states, np.zeros((n, self.model.k_states)))

    def test_simulate_1(self):
        n = 10
        measurement_shocks = np.reshape(np.arange(n * self.model.k_endog) / 10.0, (n, self.model.k_endog))
        state_shocks = np.zeros((n, self.model.ssm.k_posdef))
        initial_state = np.zeros(self.model.k_states)
        obs, states = self.model.ssm.simulate(nsimulations=n, measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
        assert_allclose(obs, np.reshape(np.arange(n * self.model.k_endog) / 10.0, (n, self.model.k_endog)))
        assert_allclose(states, np.zeros((n, self.model.k_states)))

    def test_simulate_2(self):
        n = 10
        Z = self.model['design']
        T = self.model['transition']
        measurement_shocks = np.zeros((n, self.model.k_endog))
        state_shocks = np.ones((n, self.model.ssm.k_posdef))
        initial_state = np.ones(self.model.k_states) * 2.5
        obs, states = self.model.ssm.simulate(nsimulations=n, measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
        desired_obs = np.zeros((n, self.model.k_endog))
        desired_state = np.zeros((n, self.model.k_states))
        desired_state[0] = initial_state
        desired_obs[0] = np.dot(Z, initial_state)
        for i in range(1, n):
            desired_state[i] = np.dot(T, desired_state[i - 1]) + state_shocks[i]
            desired_obs[i] = np.dot(Z, desired_state[i])
        assert_allclose(obs, desired_obs)
        assert_allclose(states, desired_state)

    def test_simulation_smoothing_0(self):
        sim = self.sim
        Z = self.model['design']
        nobs = self.model.nobs
        k_endog = self.model.k_endog
        k_posdef = self.model.ssm.k_posdef
        k_states = self.model.k_states
        sim.simulate(measurement_disturbance_variates=np.zeros(nobs * k_endog), state_disturbance_variates=np.zeros(nobs * k_posdef), initial_state_variates=np.zeros(k_states))
        assert_allclose(sim.generated_measurement_disturbance, 0)
        assert_allclose(sim.generated_state_disturbance, 0)
        assert_allclose(sim.generated_state, 0)
        assert_allclose(sim.generated_obs, 0)
        assert_allclose(sim.simulated_state, self.results.smoothed_state)
        if not self.model.ssm.filter_collapsed:
            assert_allclose(sim.simulated_measurement_disturbance, self.results.smoothed_measurement_disturbance)
        assert_allclose(sim.simulated_state_disturbance, self.results.smoothed_state_disturbance)
        if self.test_against_KFAS:
            path = os.path.join(current_path, 'results', 'results_simulation_smoothing0.csv')
            true = pd.read_csv(path)
            assert_allclose(sim.simulated_state, true[['state1', 'state2', 'state3']].T, atol=1e-07)
            assert_allclose(sim.simulated_measurement_disturbance, true[['eps1', 'eps2', 'eps3']].T, atol=1e-07)
            assert_allclose(sim.simulated_state_disturbance, true[['eta1', 'eta2', 'eta3']].T, atol=1e-07)
            signals = np.zeros((3, self.model.nobs))
            for t in range(self.model.nobs):
                signals[:, t] = np.dot(Z, sim.simulated_state[:, t])
            assert_allclose(signals, true[['signal1', 'signal2', 'signal3']].T, atol=1e-07)

    def test_simulation_smoothing_1(self):
        sim = self.sim
        Z = self.model['design']
        nobs = self.model.nobs
        k_endog = self.model.k_endog
        k_posdef = self.model.ssm.k_posdef
        k_states = self.model.k_states
        measurement_disturbance_variates = np.reshape(np.arange(nobs * k_endog) / 10.0, (nobs, k_endog))
        state_disturbance_variates = np.zeros(nobs * k_posdef)
        generated_measurement_disturbance = np.zeros(measurement_disturbance_variates.shape)
        chol = np.linalg.cholesky(self.model['obs_cov'])
        for t in range(self.model.nobs):
            generated_measurement_disturbance[t] = np.dot(chol, measurement_disturbance_variates[t])
        y = generated_measurement_disturbance.copy()
        y[np.isnan(self.model.endog)] = np.nan
        generated_model = mlemodel.MLEModel(y, k_states=k_states, k_posdef=k_posdef)
        for name in ['design', 'obs_cov', 'transition', 'selection', 'state_cov']:
            generated_model[name] = self.model[name]
        generated_model.initialize_approximate_diffuse(1000000.0)
        generated_model.ssm.filter_univariate = True
        generated_res = generated_model.ssm.smooth()
        simulated_state = 0 - generated_res.smoothed_state + self.results.smoothed_state
        if not self.model.ssm.filter_collapsed:
            simulated_measurement_disturbance = generated_measurement_disturbance.T - generated_res.smoothed_measurement_disturbance + self.results.smoothed_measurement_disturbance
        simulated_state_disturbance = 0 - generated_res.smoothed_state_disturbance + self.results.smoothed_state_disturbance
        sim.simulate(measurement_disturbance_variates=measurement_disturbance_variates, state_disturbance_variates=state_disturbance_variates, initial_state_variates=np.zeros(k_states))
        assert_allclose(sim.generated_measurement_disturbance, generated_measurement_disturbance)
        assert_allclose(sim.generated_state_disturbance, 0)
        assert_allclose(sim.generated_state, 0)
        assert_allclose(sim.generated_obs, generated_measurement_disturbance.T)
        assert_allclose(sim.simulated_state, simulated_state)
        if not self.model.ssm.filter_collapsed:
            assert_allclose(sim.simulated_measurement_disturbance, simulated_measurement_disturbance)
        assert_allclose(sim.simulated_state_disturbance, simulated_state_disturbance)
        if self.test_against_KFAS:
            path = os.path.join(current_path, 'results', 'results_simulation_smoothing1.csv')
            true = pd.read_csv(path)
            assert_allclose(sim.simulated_state, true[['state1', 'state2', 'state3']].T, atol=1e-07)
            assert_allclose(sim.simulated_measurement_disturbance, true[['eps1', 'eps2', 'eps3']].T, atol=1e-07)
            assert_allclose(sim.simulated_state_disturbance, true[['eta1', 'eta2', 'eta3']].T, atol=1e-07)
            signals = np.zeros((3, self.model.nobs))
            for t in range(self.model.nobs):
                signals[:, t] = np.dot(Z, sim.simulated_state[:, t])
            assert_allclose(signals, true[['signal1', 'signal2', 'signal3']].T, atol=1e-07)

    def test_simulation_smoothing_2(self):
        sim = self.sim
        Z = self.model['design']
        T = self.model['transition']
        nobs = self.model.nobs
        k_endog = self.model.k_endog
        k_posdef = self.model.ssm.k_posdef
        k_states = self.model.k_states
        measurement_disturbance_variates = np.reshape(np.arange(nobs * k_endog) / 10.0, (nobs, k_endog))
        state_disturbance_variates = np.reshape(np.arange(nobs * k_posdef) / 10.0, (nobs, k_posdef))
        initial_state_variates = np.zeros(k_states)
        generated_measurement_disturbance = np.zeros(measurement_disturbance_variates.shape)
        chol = np.linalg.cholesky(self.model['obs_cov'])
        for t in range(self.model.nobs):
            generated_measurement_disturbance[t] = np.dot(chol, measurement_disturbance_variates[t])
        generated_state_disturbance = np.zeros(state_disturbance_variates.shape)
        chol = np.linalg.cholesky(self.model['state_cov'])
        for t in range(self.model.nobs):
            generated_state_disturbance[t] = np.dot(chol, state_disturbance_variates[t])
        generated_obs = np.zeros((self.model.k_endog, self.model.nobs))
        generated_state = np.zeros((self.model.k_states, self.model.nobs + 1))
        chol = np.linalg.cholesky(self.results.initial_state_cov)
        generated_state[:, 0] = self.results.initial_state + np.dot(chol, initial_state_variates)
        for t in range(self.model.nobs):
            generated_state[:, t + 1] = np.dot(T, generated_state[:, t]) + generated_state_disturbance.T[:, t]
            generated_obs[:, t] = np.dot(Z, generated_state[:, t]) + generated_measurement_disturbance.T[:, t]
        y = generated_obs.copy().T
        y[np.isnan(self.model.endog)] = np.nan
        generated_model = mlemodel.MLEModel(y, k_states=k_states, k_posdef=k_posdef)
        for name in ['design', 'obs_cov', 'transition', 'selection', 'state_cov']:
            generated_model[name] = self.model[name]
        generated_model.initialize_approximate_diffuse(1000000.0)
        generated_model.ssm.filter_univariate = True
        generated_res = generated_model.ssm.smooth()
        simulated_state = generated_state[:, :-1] - generated_res.smoothed_state + self.results.smoothed_state
        if not self.model.ssm.filter_collapsed:
            simulated_measurement_disturbance = generated_measurement_disturbance.T - generated_res.smoothed_measurement_disturbance + self.results.smoothed_measurement_disturbance
        simulated_state_disturbance = generated_state_disturbance.T - generated_res.smoothed_state_disturbance + self.results.smoothed_state_disturbance
        sim.simulate(measurement_disturbance_variates=measurement_disturbance_variates, state_disturbance_variates=state_disturbance_variates, initial_state_variates=np.zeros(k_states))
        assert_allclose(sim.generated_measurement_disturbance, generated_measurement_disturbance)
        assert_allclose(sim.generated_state_disturbance, generated_state_disturbance)
        assert_allclose(sim.generated_state, generated_state)
        assert_allclose(sim.generated_obs, generated_obs)
        assert_allclose(sim.simulated_state, simulated_state, atol=1e-07)
        if not self.model.ssm.filter_collapsed:
            assert_allclose(sim.simulated_measurement_disturbance.T, simulated_measurement_disturbance.T)
        assert_allclose(sim.simulated_state_disturbance, simulated_state_disturbance)
        if self.test_against_KFAS:
            path = os.path.join(current_path, 'results', 'results_simulation_smoothing2.csv')
            true = pd.read_csv(path)
            assert_allclose(sim.simulated_state.T, true[['state1', 'state2', 'state3']], atol=1e-07)
            assert_allclose(sim.simulated_measurement_disturbance, true[['eps1', 'eps2', 'eps3']].T, atol=1e-07)
            assert_allclose(sim.simulated_state_disturbance, true[['eta1', 'eta2', 'eta3']].T, atol=1e-07)
            signals = np.zeros((3, self.model.nobs))
            for t in range(self.model.nobs):
                signals[:, t] = np.dot(Z, sim.simulated_state[:, t])
            assert_allclose(signals, true[['signal1', 'signal2', 'signal3']].T, atol=1e-07)