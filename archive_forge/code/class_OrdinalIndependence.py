from statsmodels.compat.pandas import Appender
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as spl
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import bool_like
class OrdinalIndependence(CategoricalCovStruct):
    """
    An independence covariance structure for ordinal models.

    The working covariance between indicators derived from different
    observations is zero.  The working covariance between indicators
    derived form a common observation is determined from their current
    mean values.

    There are no parameters to estimate in this covariance structure.
    """

    def covariance_matrix(self, expected_value, index):
        ibd = self.ibd[index]
        n = len(expected_value)
        vmat = np.zeros((n, n))
        for bdl in ibd:
            ev = expected_value[bdl[0]:bdl[1]]
            vmat[bdl[0]:bdl[1], bdl[0]:bdl[1]] = np.minimum.outer(ev, ev) - np.outer(ev, ev)
        return (vmat, False)

    def update(self, params):
        pass