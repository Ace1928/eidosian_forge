from abc import ABCMeta, abstractmethod
from numbers import Integral
import numpy as np
import scipy.sparse as sp
from .base import (
from .model_selection import cross_val_predict
from .utils import Bunch, _print_elapsed_time, check_random_state
from .utils._param_validation import HasMethods, StrOptions
from .utils.metadata_routing import (
from .utils.metaestimators import available_if
from .utils.multiclass import check_classification_targets
from .utils.parallel import Parallel, delayed
from .utils.validation import _check_method_params, check_is_fitted, has_fit_parameter
def _partial_fit_estimator(estimator, X, y, classes=None, partial_fit_params=None, first_time=True):
    partial_fit_params = {} if partial_fit_params is None else partial_fit_params
    if first_time:
        estimator = clone(estimator)
    if classes is not None:
        estimator.partial_fit(X, y, classes=classes, **partial_fit_params)
    else:
        estimator.partial_fit(X, y, **partial_fit_params)
    return estimator