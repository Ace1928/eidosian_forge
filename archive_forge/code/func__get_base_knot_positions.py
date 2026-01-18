import collections
from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r
from numbers import Integral
import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.special import comb
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import check_array
from ..utils._param_validation import Interval, StrOptions
from ..utils.fixes import parse_version, sp_version
from ..utils.stats import _weighted_percentile
from ..utils.validation import (
from ._csr_polynomial_expansion import (
@staticmethod
def _get_base_knot_positions(X, n_knots=10, knots='uniform', sample_weight=None):
    """Calculate base knot positions.

        Base knots such that first knot <= feature <= last knot. For the
        B-spline construction with scipy.interpolate.BSpline, 2*degree knots
        beyond the base interval are added.

        Returns
        -------
        knots : ndarray of shape (n_knots, n_features), dtype=np.float64
            Knot positions (points) of base interval.
        """
    if knots == 'quantile':
        percentiles = 100 * np.linspace(start=0, stop=1, num=n_knots, dtype=np.float64)
        if sample_weight is None:
            knots = np.percentile(X, percentiles, axis=0)
        else:
            knots = np.array([_weighted_percentile(X, sample_weight, percentile) for percentile in percentiles])
    else:
        mask = slice(None, None, 1) if sample_weight is None else sample_weight > 0
        x_min = np.amin(X[mask], axis=0)
        x_max = np.amax(X[mask], axis=0)
        knots = np.linspace(start=x_min, stop=x_max, num=n_knots, endpoint=True, dtype=np.float64)
    return knots