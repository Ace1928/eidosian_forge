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
def _translate_train_sizes(train_sizes, n_max_training_samples):
    """Determine absolute sizes of training subsets and validate 'train_sizes'.

    Examples:
        _translate_train_sizes([0.5, 1.0], 10) -> [5, 10]
        _translate_train_sizes([5, 10], 10) -> [5, 10]

    Parameters
    ----------
    train_sizes : array-like of shape (n_ticks,)
        Numbers of training examples that will be used to generate the
        learning curve. If the dtype is float, it is regarded as a
        fraction of 'n_max_training_samples', i.e. it has to be within (0, 1].

    n_max_training_samples : int
        Maximum number of training samples (upper bound of 'train_sizes').

    Returns
    -------
    train_sizes_abs : array of shape (n_unique_ticks,)
        Numbers of training examples that will be used to generate the
        learning curve. Note that the number of ticks might be less
        than n_ticks because duplicate entries will be removed.
    """
    train_sizes_abs = np.asarray(train_sizes)
    n_ticks = train_sizes_abs.shape[0]
    n_min_required_samples = np.min(train_sizes_abs)
    n_max_required_samples = np.max(train_sizes_abs)
    if np.issubdtype(train_sizes_abs.dtype, np.floating):
        if n_min_required_samples <= 0.0 or n_max_required_samples > 1.0:
            raise ValueError('train_sizes has been interpreted as fractions of the maximum number of training samples and must be within (0, 1], but is within [%f, %f].' % (n_min_required_samples, n_max_required_samples))
        train_sizes_abs = (train_sizes_abs * n_max_training_samples).astype(dtype=int, copy=False)
        train_sizes_abs = np.clip(train_sizes_abs, 1, n_max_training_samples)
    elif n_min_required_samples <= 0 or n_max_required_samples > n_max_training_samples:
        raise ValueError('train_sizes has been interpreted as absolute numbers of training samples and must be within (0, %d], but is within [%d, %d].' % (n_max_training_samples, n_min_required_samples, n_max_required_samples))
    train_sizes_abs = np.unique(train_sizes_abs)
    if n_ticks > train_sizes_abs.shape[0]:
        warnings.warn("Removed duplicate entries from 'train_sizes'. Number of ticks will be less than the size of 'train_sizes': %d instead of %d." % (train_sizes_abs.shape[0], n_ticks), RuntimeWarning)
    return train_sizes_abs