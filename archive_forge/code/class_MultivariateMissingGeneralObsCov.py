import os
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.tests.results import results_kalman_filter
from statsmodels.tsa.statespace.sarimax import SARIMAX
class MultivariateMissingGeneralObsCov:

    @classmethod
    def setup_class(cls, which, dtype=float, alternate_timing=False, **kwargs):
        path = os.path.join(current_path, 'results', 'results_smoothing_generalobscov_R.csv')
        cls.desired = pd.read_csv(path)
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start='1959-01-01', end='2009-7-01', freq='QS')
        obs = dta[['realgdp', 'realcons', 'realinv']].diff().iloc[1:]
        if which == 'all':
            obs.iloc[:50, :] = np.nan
            obs.iloc[119:130, :] = np.nan
        elif which == 'partial':
            obs.iloc[0:50, 0] = np.nan
            obs.iloc[119:130, 0] = np.nan
        elif which == 'mixed':
            obs.iloc[0:50, 0] = np.nan
            obs.iloc[19:70, 1] = np.nan
            obs.iloc[39:90, 2] = np.nan
            obs.iloc[119:130, 0] = np.nan
            obs.iloc[119:130, 2] = np.nan
        mod = MLEModel(obs, k_states=3, k_posdef=3, **kwargs)
        mod['design'] = np.eye(3)
        X = (np.arange(9) + 1).reshape((3, 3)) / 10.0
        mod['obs_cov'] = np.dot(X, X.T)
        mod['transition'] = np.eye(3)
        mod['selection'] = np.eye(3)
        mod['state_cov'] = np.eye(3)
        mod.initialize_approximate_diffuse(1000000.0)
        cls.model = mod.ssm
        cls.model.filter_conventional = True
        cls.conventional_results = cls.model.smooth()
        n_disturbance_variates = (cls.model.k_endog + cls.model.k_posdef) * cls.model.nobs
        cls.conventional_sim = cls.model.simulation_smoother(disturbance_variates=np.zeros(n_disturbance_variates), initial_state_variates=np.zeros(cls.model.k_states))
        cls.model.filter_univariate = True
        cls.univariate_results = cls.model.smooth()
        cls.univariate_sim = cls.model.simulation_smoother(disturbance_variates=np.zeros(n_disturbance_variates), initial_state_variates=np.zeros(cls.model.k_states))

    def test_using_univariate(self):
        assert not self.conventional_results.filter_univariate
        assert self.univariate_results.filter_univariate
        assert_allclose(self.conventional_results.forecasts_error_cov[1, 1, 0], 1000000.77)
        assert_allclose(self.univariate_results.forecasts_error_cov[1, 1, 0], 1000000.77)

    def test_forecasts(self):
        assert_almost_equal(self.conventional_results.forecasts[0, :], self.univariate_results.forecasts[0, :], 9)

    def test_forecasts_error(self):
        assert_almost_equal(self.conventional_results.forecasts_error[0, :], self.univariate_results.forecasts_error[0, :], 9)

    def test_forecasts_error_cov(self):
        assert_almost_equal(self.conventional_results.forecasts_error_cov[0, 0, :], self.univariate_results.forecasts_error_cov[0, 0, :], 9)

    def test_filtered_state(self):
        assert_almost_equal(self.conventional_results.filtered_state, self.univariate_results.filtered_state, 8)

    def test_filtered_state_cov(self):
        assert_almost_equal(self.conventional_results.filtered_state_cov, self.univariate_results.filtered_state_cov, 9)

    def test_predicted_state(self):
        assert_almost_equal(self.conventional_results.predicted_state, self.univariate_results.predicted_state, 8)

    def test_predicted_state_cov(self):
        assert_almost_equal(self.conventional_results.predicted_state_cov, self.univariate_results.predicted_state_cov, 9)

    def test_loglike(self):
        assert_allclose(self.conventional_results.llf_obs, self.univariate_results.llf_obs)

    def test_smoothed_states(self):
        assert_almost_equal(self.conventional_results.smoothed_state, self.univariate_results.smoothed_state, 7)

    def test_smoothed_states_cov(self):
        assert_almost_equal(self.conventional_results.smoothed_state_cov, self.univariate_results.smoothed_state_cov, 6)

    @pytest.mark.skip
    def test_smoothed_measurement_disturbance(self):
        assert_almost_equal(self.conventional_results.smoothed_measurement_disturbance, self.univariate_results.smoothed_measurement_disturbance, 9)

    @pytest.mark.skip
    def test_smoothed_measurement_disturbance_cov(self):
        conv = self.conventional_results
        univ = self.univariate_results
        assert_almost_equal(conv.smoothed_measurement_disturbance_cov.diagonal(), univ.smoothed_measurement_disturbance_cov.diagonal(), 9)

    def test_smoothed_state_disturbance(self):
        assert_allclose(self.conventional_results.smoothed_state_disturbance, self.univariate_results.smoothed_state_disturbance, atol=1e-07)

    def test_smoothed_state_disturbance_cov(self):
        assert_almost_equal(self.conventional_results.smoothed_state_disturbance_cov, self.univariate_results.smoothed_state_disturbance_cov, 9)

    def test_simulation_smoothed_state(self):
        assert_almost_equal(self.conventional_sim.simulated_state, self.univariate_sim.simulated_state, 9)

    @pytest.mark.skip
    def test_simulation_smoothed_measurement_disturbance(self):
        assert_almost_equal(self.conventional_sim.simulated_measurement_disturbance, self.univariate_sim.simulated_measurement_disturbance, 9)

    def test_simulation_smoothed_state_disturbance(self):
        assert_almost_equal(self.conventional_sim.simulated_state_disturbance, self.univariate_sim.simulated_state_disturbance, 9)