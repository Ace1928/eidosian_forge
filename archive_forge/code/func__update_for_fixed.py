from statsmodels.compat.pandas import deprecate_kwarg
import contextlib
from typing import Any
from collections.abc import Hashable, Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, least_squares, minimize
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from statsmodels.tools.validation import (
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.exponential_smoothing.ets import (
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import HoltWintersArgs
from statsmodels.tsa.holtwinters._smoothers import (
from statsmodels.tsa.holtwinters.results import (
from statsmodels.tsa.tsatools import freq_to_period
def _update_for_fixed(self, sel, alpha, beta, gamma, phi, l0, b0, s0):
    if self._fixed_parameters:
        fixed = self._fixed_parameters
        names = self._ordered_names()
        not_fixed = np.array([name not in fixed for name in names])
        if (~sel[~not_fixed]).any():
            invalid = []
            for name, s, nf in zip(names, sel, not_fixed):
                if not s and (not nf):
                    invalid.append(name)
            invalid_names = ', '.join(invalid)
            raise ValueError(f'Cannot fix a parameter that is not being estimated: {invalid_names}')
        sel &= not_fixed
        alpha = fixed.get('smoothing_level', alpha)
        beta = fixed.get('smoothing_trend', beta)
        gamma = fixed.get('smoothing_seasonal', gamma)
        phi = fixed.get('damping_trend', phi)
        l0 = fixed.get('initial_level', l0)
        b0 = fixed.get('initial_trend', b0)
        for i in range(self.seasonal_periods):
            s0[i] = fixed.get(f'initial_seasonal.{i}', s0[i])
    return (sel, alpha, beta, gamma, phi, l0, b0, s0)