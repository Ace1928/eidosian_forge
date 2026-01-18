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
def _predict_binary(estimator, X):
    """Make predictions using a single binary estimator."""
    if is_regressor(estimator):
        return estimator.predict(X)
    try:
        score = np.ravel(estimator.decision_function(X))
    except (AttributeError, NotImplementedError):
        score = estimator.predict_proba(X)[:, 1]
    return score