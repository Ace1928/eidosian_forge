import warnings
from numbers import Integral
import numpy as np
from sklearn.neighbors._base import _check_precomputed
from ..base import ClassifierMixin, _fit_context
from ..metrics._pairwise_distances_reduction import (
from ..utils._param_validation import StrOptions
from ..utils.arrayfuncs import _all_with_any_reduction_axis_1
from ..utils.extmath import weighted_mode
from ..utils.fixes import _mode
from ..utils.validation import _is_arraylike, _num_samples, check_is_fitted
from ._base import KNeighborsMixin, NeighborsBase, RadiusNeighborsMixin, _get_weights
Return probability estimates for the test data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_queries, n_features),                 or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        p : ndarray of shape (n_queries, n_classes), or a list of                 n_outputs of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        