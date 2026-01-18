import array
import itertools
import warnings
from collections import defaultdict
from numbers import Integral
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import column_or_1d
from ..utils._encode import _encode, _unique
from ..utils._param_validation import Interval, validate_params
from ..utils.multiclass import type_of_target, unique_labels
from ..utils.sparsefuncs import min_max_axis
from ..utils.validation import _num_samples, check_array, check_is_fitted
Transform the given indicator matrix into label sets.

        Parameters
        ----------
        yt : {ndarray, sparse matrix} of shape (n_samples, n_classes)
            A matrix containing only 1s ands 0s.

        Returns
        -------
        y : list of tuples
            The set of labels for each sample such that `y[i]` consists of
            `classes_[j]` for each `yt[i, j] == 1`.
        