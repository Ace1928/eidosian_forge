import threading
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from warnings import catch_warnings, simplefilter, warn
import numpy as np
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import issparse
from ..base import (
from ..exceptions import DataConversionWarning
from ..metrics import accuracy_score, r2_score
from ..preprocessing import OneHotEncoder
from ..tree import (
from ..tree._tree import DOUBLE, DTYPE
from ..utils import check_random_state, compute_sample_weight
from ..utils._param_validation import Interval, RealNotInt, StrOptions
from ..utils._tags import _safe_tags
from ..utils.multiclass import check_classification_targets, type_of_target
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from ._base import BaseEnsemble, _partition_estimators
def _parallel_build_trees(tree, bootstrap, X, y, sample_weight, tree_idx, n_trees, verbose=0, class_weight=None, n_samples_bootstrap=None, missing_values_in_feature_mask=None):
    """
    Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print('building tree %d of %d' % (tree_idx + 1, n_trees))
    if bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()
        indices = _generate_sample_indices(tree.random_state, n_samples, n_samples_bootstrap)
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts
        if class_weight == 'subsample':
            with catch_warnings():
                simplefilter('ignore', DeprecationWarning)
                curr_sample_weight *= compute_sample_weight('auto', y, indices=indices)
        elif class_weight == 'balanced_subsample':
            curr_sample_weight *= compute_sample_weight('balanced', y, indices=indices)
        tree._fit(X, y, sample_weight=curr_sample_weight, check_input=False, missing_values_in_feature_mask=missing_values_in_feature_mask)
    else:
        tree._fit(X, y, sample_weight=sample_weight, check_input=False, missing_values_in_feature_mask=missing_values_in_feature_mask)
    return tree