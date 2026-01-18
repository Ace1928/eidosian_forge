import numpy as np
from types import SimpleNamespace
from statsmodels.tsa.statespace.representation import OptionWrapper
from statsmodels.tsa.statespace.kalman_filter import (KalmanFilter,
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.statespace import tools, initialization
def _initialize_smoother(self, smoother_output=None, smooth_method=None, prefix=None, **kwargs):
    if smoother_output is None:
        smoother_output = self.smoother_output
    if smooth_method is None:
        smooth_method = self.smooth_method
    prefix, dtype, create_filter, create_statespace = self._initialize_filter(prefix, **kwargs)
    create_smoother = create_filter or prefix not in self._kalman_smoothers
    if not create_smoother:
        kalman_smoother = self._kalman_smoothers[prefix]
        create_smoother = kalman_smoother.kfilter is not self._kalman_filters[prefix]
    if create_smoother:
        cls = self.prefix_kalman_smoother_map[prefix]
        self._kalman_smoothers[prefix] = cls(self._statespaces[prefix], self._kalman_filters[prefix], smoother_output, smooth_method)
    else:
        self._kalman_smoothers[prefix].set_smoother_output(smoother_output, False)
        self._kalman_smoothers[prefix].set_smooth_method(smooth_method)
    return (prefix, dtype, create_smoother, create_filter, create_statespace)