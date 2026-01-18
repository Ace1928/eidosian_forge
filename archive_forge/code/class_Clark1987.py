import os
import warnings
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace.representation import Representation
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.simulation_smoother import SimulationSmoother
from statsmodels.tsa.statespace import tools, sarimax
from .results import results_kalman_filter
from numpy.testing import (
class Clark1987:
    """
    Clark's (1987) univariate unobserved components model of real GDP (as
    presented in Kim and Nelson, 1999)

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `results.results_kalman_filter` for more information.
    """

    @classmethod
    def setup_class(cls, dtype=float, **kwargs):
        cls.true = results_kalman_filter.uc_uni
        cls.true_states = pd.DataFrame(cls.true['states'])
        data = pd.DataFrame(cls.true['data'], index=pd.date_range('1947-01-01', '1995-07-01', freq='QS'), columns=['GDP'])
        data['lgdp'] = np.log(data['GDP'])
        k_states = 4
        cls.model = KalmanFilter(k_endog=1, k_states=k_states, **kwargs)
        cls.model.bind(data['lgdp'].values)
        cls.model.design[:, :, 0] = [1, 1, 0, 0]
        cls.model.transition[[0, 0, 1, 1, 2, 3], [0, 3, 1, 2, 1, 3], [0, 0, 0, 0, 0, 0]] = [1, 1, 0, 0, 1, 1]
        cls.model.selection = np.eye(cls.model.k_states)
        sigma_v, sigma_e, sigma_w, phi_1, phi_2 = np.array(cls.true['parameters'])
        cls.model.transition[[1, 1], [1, 2], [0, 0]] = [phi_1, phi_2]
        cls.model.state_cov[np.diag_indices(k_states) + (np.zeros(k_states, dtype=int),)] = [sigma_v ** 2, sigma_e ** 2, 0, sigma_w ** 2]
        initial_state = np.zeros((k_states,))
        initial_state_cov = np.eye(k_states) * 100
        initial_state_cov = np.dot(np.dot(cls.model.transition[:, :, 0], initial_state_cov), cls.model.transition[:, :, 0].T)
        cls.model.initialize_known(initial_state, initial_state_cov)

    @classmethod
    def run_filter(cls):
        return cls.model.filter()

    def test_loglike(self):
        assert_almost_equal(self.results.llf_obs[self.true['start']:].sum(), self.true['loglike'], 5)

    def test_filtered_state(self):
        assert_almost_equal(self.results.filtered_state[0][self.true['start']:], self.true_states.iloc[:, 0], 4)
        assert_almost_equal(self.results.filtered_state[1][self.true['start']:], self.true_states.iloc[:, 1], 4)
        assert_almost_equal(self.results.filtered_state[3][self.true['start']:], self.true_states.iloc[:, 2], 4)