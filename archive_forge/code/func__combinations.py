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
def _combinations(n_features, min_degree, max_degree, interaction_only, include_bias):
    comb = combinations if interaction_only else combinations_w_r
    start = max(1, min_degree)
    iter = chain.from_iterable((comb(range(n_features), i) for i in range(start, max_degree + 1)))
    if include_bias:
        iter = chain(comb(range(n_features), 0), iter)
    return iter