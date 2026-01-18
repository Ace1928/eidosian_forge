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
class TestStatesMissingAR3:

    @classmethod
    def setup_class(cls, alternate_timing=False, *args, **kwargs):
        path = os.path.join(current_path, 'results', 'results_wpi1_ar3_stata.csv')
        cls.stata = pd.read_csv(path)
        cls.stata.index = pd.date_range(start='1960-01-01', periods=124, freq='QS')
        path = os.path.join(current_path, 'results', 'results_wpi1_missing_ar3_matlab_ssm.csv')
        matlab_names = ['a1', 'a2', 'a3', 'detP', 'alphahat1', 'alphahat2', 'alphahat3', 'detV', 'eps', 'epsvar', 'eta', 'etavar']
        cls.matlab_ssm = pd.read_csv(path, header=None, names=matlab_names)
        path = os.path.join(current_path, 'results', 'results_smoothing3_R.csv')
        cls.R_ssm = pd.read_csv(path)
        cls.stata['dwpi'] = cls.stata['wpi'].diff()
        cls.stata.loc[cls.stata.index[10:21], 'dwpi'] = np.nan
        cls.model = sarimax.SARIMAX(cls.stata.loc[cls.stata.index[1:], 'dwpi'], *args, order=(3, 0, 0), hamilton_representation=True, **kwargs)
        if alternate_timing:
            cls.model.ssm.timing_init_filtered = True
        params = np.r_[0.5270715, 0.0952613, 0.2580355, 0.5307459]
        cls.results = cls.model.smooth(params, return_ssm=True)
        cls.results.det_predicted_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_cov = np.zeros((1, cls.model.nobs))
        for i in range(cls.model.nobs):
            cls.results.det_predicted_state_cov[0, i] = np.linalg.det(cls.results.predicted_state_cov[:, :, i])
            cls.results.det_smoothed_state_cov[0, i] = np.linalg.det(cls.results.smoothed_state_cov[:, :, i])
        nobs = cls.model.nobs
        k_endog = cls.model.k_endog
        k_posdef = cls.model.ssm.k_posdef
        cls.sim = cls.model.simulation_smoother()
        cls.sim.simulate(measurement_disturbance_variates=np.zeros(nobs * k_endog), state_disturbance_variates=np.zeros(nobs * k_posdef), initial_state_variates=np.zeros(cls.model.k_states))

    def test_predicted_states(self):
        assert_almost_equal(self.results.predicted_state[:, :-1].T, self.matlab_ssm[['a1', 'a2', 'a3']], 4)

    def test_predicted_states_cov(self):
        assert_almost_equal(self.results.det_predicted_state_cov.T, self.matlab_ssm[['detP']], 4)

    def test_smoothed_states(self):
        assert_almost_equal(self.results.smoothed_state.T, self.matlab_ssm[['alphahat1', 'alphahat2', 'alphahat3']], 4)

    def test_smoothed_states_cov(self):
        assert_almost_equal(self.results.det_smoothed_state_cov.T, self.matlab_ssm[['detV']], 4)

    def test_smoothed_measurement_disturbance(self):
        assert_almost_equal(self.results.smoothed_measurement_disturbance.T, self.matlab_ssm[['eps']], 4)

    def test_smoothed_measurement_disturbance_cov(self):
        assert_almost_equal(self.results.smoothed_measurement_disturbance_cov[0].T, self.matlab_ssm[['epsvar']], 4)

    def test_smoothed_state_disturbance(self):
        assert_almost_equal(self.results.smoothed_state_disturbance.T, self.R_ssm[['etahat']], 9)

    def test_smoothed_state_disturbance_cov(self):
        assert_almost_equal(self.results.smoothed_state_disturbance_cov[0, 0, :], self.R_ssm['detVeta'], 9)