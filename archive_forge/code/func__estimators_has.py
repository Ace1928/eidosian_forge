import array
import itertools
import warnings
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from .base import (
from .metrics.pairwise import pairwise_distances_argmin
from .preprocessing import LabelBinarizer
from .utils import check_random_state
from .utils._param_validation import HasMethods, Interval
from .utils._tags import _safe_tags
from .utils.metadata_routing import (
from .utils.metaestimators import _safe_split, available_if
from .utils.multiclass import (
from .utils.parallel import Parallel, delayed
from .utils.validation import _check_method_params, _num_samples, check_is_fitted
def _estimators_has(attr):
    """Check if self.estimator or self.estimators_[0] has attr.

    If `self.estimators_[0]` has the attr, then its safe to assume that other
    estimators have it too. We raise the original `AttributeError` if `attr`
    does not exist. This function is used together with `available_if`.
    """

    def check(self):
        if hasattr(self, 'estimators_'):
            getattr(self.estimators_[0], attr)
        else:
            getattr(self.estimator, attr)
        return True
    return check