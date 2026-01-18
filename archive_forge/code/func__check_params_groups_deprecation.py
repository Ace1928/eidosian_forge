import numbers
import time
import warnings
from collections import Counter
from contextlib import suppress
from functools import partial
from numbers import Real
from traceback import format_exc
import numpy as np
import scipy.sparse as sp
from joblib import logger
from ..base import clone, is_classifier
from ..exceptions import FitFailedWarning, UnsetMetadataPassedError
from ..metrics import check_scoring, get_scorer_names
from ..metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from ..preprocessing import LabelEncoder
from ..utils import Bunch, _safe_indexing, check_random_state, indexable
from ..utils._param_validation import (
from ..utils.metadata_routing import (
from ..utils.metaestimators import _safe_split
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _check_method_params, _num_samples
from ._split import check_cv
def _check_params_groups_deprecation(fit_params, params, groups):
    """A helper function to check deprecations on `groups` and `fit_params`.

    To be removed when set_config(enable_metadata_routing=False) is not possible.
    """
    if params is not None and fit_params is not None:
        raise ValueError('`params` and `fit_params` cannot both be provided. Pass parameters via `params`. `fit_params` is deprecated and will be removed in version 1.6.')
    elif fit_params is not None:
        warnings.warn('`fit_params` is deprecated and will be removed in version 1.6. Pass parameters via `params` instead.', FutureWarning)
        params = fit_params
    params = {} if params is None else params
    if groups is not None and _routing_enabled():
        raise ValueError('`groups` can only be passed if metadata routing is not enabled via `sklearn.set_config(enable_metadata_routing=True)`. When routing is enabled, pass `groups` alongside other metadata via the `params` argument instead.')
    return params