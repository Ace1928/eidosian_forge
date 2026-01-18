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
def _set_oob_score(self, X, y):
    raise NotImplementedError('OOB score not supported by iforest')