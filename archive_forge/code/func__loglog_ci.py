from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import warnings
import numpy as np
from scipy import special, interpolate, stats
from scipy.stats._censored_data import CensoredData
from scipy.stats._common import ConfidenceInterval
def _loglog_ci(self, confidence_level):
    sf, d, n = (self._sf, self._d, self._n)
    with np.errstate(divide='ignore', invalid='ignore'):
        var = 1 / np.log(sf) ** 2 * np.cumsum(d / (n * (n - d)))
    se = np.sqrt(var)
    z = special.ndtri(1 / 2 + confidence_level / 2)
    with np.errstate(divide='ignore'):
        lnl_points = np.log(-np.log(sf))
    z_se = z * se
    low = np.exp(-np.exp(lnl_points + z_se))
    high = np.exp(-np.exp(lnl_points - z_se))
    if self._kind == 'cdf':
        low, high = (1 - high, 1 - low)
    return (low, high)