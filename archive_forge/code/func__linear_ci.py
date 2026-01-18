from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import warnings
import numpy as np
from scipy import special, interpolate, stats
from scipy.stats._censored_data import CensoredData
from scipy.stats._common import ConfidenceInterval
def _linear_ci(self, confidence_level):
    sf, d, n = (self._sf, self._d, self._n)
    with np.errstate(divide='ignore', invalid='ignore'):
        var = sf ** 2 * np.cumsum(d / (n * (n - d)))
    se = np.sqrt(var)
    z = special.ndtri(1 / 2 + confidence_level / 2)
    z_se = z * se
    low = self.probabilities - z_se
    high = self.probabilities + z_se
    return (low, high)