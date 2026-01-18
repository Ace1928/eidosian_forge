from numbers import Number
from statistics import NormalDist
import numpy as np
import pandas as pd
from .algorithms import bootstrap
from .utils import _check_argument
def _percentile_interval(data, width):
    """Return a percentile interval from data of a given width."""
    edge = (100 - width) / 2
    percentiles = (edge, 100 - edge)
    return np.nanpercentile(data, percentiles)