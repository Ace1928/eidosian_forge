import numbers
from numbers import Integral, Real
from warnings import warn
import numpy as np
from scipy.sparse import issparse
from ..base import OutlierMixin, _fit_context
from ..tree import ExtraTreeRegressor
from ..tree._tree import DTYPE as tree_dtype
from ..utils import (
from ..utils._param_validation import Interval, RealNotInt, StrOptions
from ..utils.validation import _num_samples, check_is_fitted
from ._bagging import BaseBagging
def _score_samples(self, X):
    """Private version of score_samples without input validation.

        Input validation would remove feature names, so we disable it.
        """
    check_is_fitted(self)
    return -self._compute_chunked_score_samples(X)