import numpy as np
from types import SimpleNamespace
from statsmodels.tsa.statespace.representation import OptionWrapper
from statsmodels.tsa.statespace.kalman_filter import (KalmanFilter,
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.statespace import tools, initialization
def _get_smoothed_forecasts(self):
    if self._smoothed_forecasts is None:
        self._smoothed_forecasts = np.zeros(self.forecasts.shape, dtype=self.dtype)
        self._smoothed_forecasts_error = np.zeros(self.forecasts_error.shape, dtype=self.dtype)
        self._smoothed_forecasts_error_cov = np.zeros(self.forecasts_error_cov.shape, dtype=self.dtype)
        for t in range(self.nobs):
            design_t = 0 if self.design.shape[2] == 1 else t
            obs_cov_t = 0 if self.obs_cov.shape[2] == 1 else t
            obs_intercept_t = 0 if self.obs_intercept.shape[1] == 1 else t
            mask = ~self.missing[:, t].astype(bool)
            self._smoothed_forecasts[:, t] = np.dot(self.design[:, :, design_t], self.smoothed_state[:, t]) + self.obs_intercept[:, obs_intercept_t]
            if self.nmissing[t] > 0:
                self._smoothed_forecasts_error[:, t] = np.nan
            self._smoothed_forecasts_error[mask, t] = self.endog[mask, t] - self._smoothed_forecasts[mask, t]
            self._smoothed_forecasts_error_cov[:, :, t] = np.dot(np.dot(self.design[:, :, design_t], self.smoothed_state_cov[:, :, t]), self.design[:, :, design_t].T) + self.obs_cov[:, :, obs_cov_t]
    return (self._smoothed_forecasts, self._smoothed_forecasts_error, self._smoothed_forecasts_error_cov)