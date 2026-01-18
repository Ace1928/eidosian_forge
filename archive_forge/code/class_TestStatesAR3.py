import os
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from statsmodels.tsa.statespace.kalman_filter import FILTER_UNIVARIATE
from statsmodels.tsa.statespace.kalman_smoother import (
class TestStatesAR3:

    @classmethod
    def setup_class(cls, alternate_timing=False, *args, **kwargs):
        path = os.path.join(current_path, 'results', 'results_wpi1_ar3_stata.csv')
        cls.stata = pd.read_csv(path)
        cls.stata.index = pd.date_range(start='1960-01-01', periods=124, freq='QS')
        path = os.path.join(current_path, 'results', 'results_wpi1_ar3_matlab_ssm.csv')
        matlab_names = ['a1', 'a2', 'a3', 'detP', 'alphahat1', 'alphahat2', 'alphahat3', 'detV', 'eps', 'epsvar', 'eta', 'etavar']
        cls.matlab_ssm = pd.read_csv(path, header=None, names=matlab_names)
        cls.model = sarimax.SARIMAX(cls.stata['wpi'], *args, order=(3, 1, 0), simple_differencing=True, hamilton_representation=True, **kwargs)
        if alternate_timing:
            cls.model.ssm.timing_init_filtered = True
        params = np.r_[0.5270715, 0.0952613, 0.2580355, 0.5307459]
        cls.results = cls.model.smooth(params, cov_type='none')
        cls.results.det_predicted_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_cov = np.zeros((1, cls.model.nobs))
        for i in range(cls.model.nobs):
            cls.results.det_predicted_state_cov[0, i] = np.linalg.det(cls.results.filter_results.predicted_state_cov[:, :, i])
            cls.results.det_smoothed_state_cov[0, i] = np.linalg.det(cls.results.smoother_results.smoothed_state_cov[:, :, i])
        nobs = cls.model.nobs
        k_endog = cls.model.k_endog
        k_posdef = cls.model.ssm.k_posdef
        cls.sim = cls.model.simulation_smoother(filter_timing=0)
        cls.sim.simulate(measurement_disturbance_variates=np.zeros(nobs * k_endog), state_disturbance_variates=np.zeros(nobs * k_posdef), initial_state_variates=np.zeros(cls.model.k_states))

    def test_predict_obs(self):
        assert_almost_equal(self.results.filter_results.predict().forecasts[0], self.stata.iloc[1:]['dep1'], 4)

    def test_standardized_residuals(self):
        assert_almost_equal(self.results.filter_results.standardized_forecasts_error[0], self.stata.iloc[1:]['sr1'], 4)

    def test_predicted_states(self):
        assert_almost_equal(self.results.filter_results.predicted_state[:, :-1].T, self.stata.iloc[1:][['sp1', 'sp2', 'sp3']], 4)
        assert_almost_equal(self.results.filter_results.predicted_state[:, :-1].T, self.matlab_ssm[['a1', 'a2', 'a3']], 4)

    def test_predicted_states_cov(self):
        assert_almost_equal(self.results.det_predicted_state_cov.T, self.matlab_ssm[['detP']], 4)

    def test_filtered_states(self):
        assert_almost_equal(self.results.filter_results.filtered_state.T, self.stata.iloc[1:][['sf1', 'sf2', 'sf3']], 4)

    def test_smoothed_states(self):
        assert_almost_equal(self.results.smoother_results.smoothed_state.T, self.stata.iloc[1:][['sm1', 'sm2', 'sm3']], 4)
        assert_almost_equal(self.results.smoother_results.smoothed_state.T, self.matlab_ssm[['alphahat1', 'alphahat2', 'alphahat3']], 4)

    def test_smoothed_states_cov(self):
        assert_almost_equal(self.results.det_smoothed_state_cov.T, self.matlab_ssm[['detV']], 4)

    def test_smoothed_measurement_disturbance(self):
        assert_almost_equal(self.results.smoother_results.smoothed_measurement_disturbance.T, self.matlab_ssm[['eps']], 4)

    def test_smoothed_measurement_disturbance_cov(self):
        res = self.results.smoother_results
        assert_almost_equal(res.smoothed_measurement_disturbance_cov[0].T, self.matlab_ssm[['epsvar']], 4)

    def test_smoothed_state_disturbance(self):
        assert_almost_equal(self.results.smoother_results.smoothed_state_disturbance.T, self.matlab_ssm[['eta']], 4)

    def test_smoothed_state_disturbance_cov(self):
        assert_almost_equal(self.results.smoother_results.smoothed_state_disturbance_cov[0].T, self.matlab_ssm[['etavar']], 4)