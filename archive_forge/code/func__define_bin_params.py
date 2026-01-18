from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
import numpy as np
import pandas as pd
from pandas import DataFrame
from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Scale
from seaborn._stats.base import Stat
from typing import TYPE_CHECKING
def _define_bin_params(self, data, orient, scale_type):
    """Given data, return numpy.histogram parameters to define bins."""
    vals = data[orient]
    weights = data.get('weight', None)
    discrete = self.discrete or scale_type == 'nominal'
    bin_edges = self._define_bin_edges(vals, weights, self.bins, self.binwidth, self.binrange, discrete)
    if isinstance(self.bins, (str, int)):
        n_bins = len(bin_edges) - 1
        bin_range = (bin_edges.min(), bin_edges.max())
        bin_kws = dict(bins=n_bins, range=bin_range)
    else:
        bin_kws = dict(bins=bin_edges)
    return bin_kws