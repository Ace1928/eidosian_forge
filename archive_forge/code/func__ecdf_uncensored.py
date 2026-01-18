from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import warnings
import numpy as np
from scipy import special, interpolate, stats
from scipy.stats._censored_data import CensoredData
from scipy.stats._common import ConfidenceInterval
def _ecdf_uncensored(sample):
    sample = np.sort(sample)
    x, counts = np.unique(sample, return_counts=True)
    events = np.cumsum(counts)
    n = sample.size
    cdf = events / n
    sf = 1 - cdf
    at_risk = np.concatenate(([n], n - events[:-1]))
    return (x, cdf, sf, at_risk, counts)