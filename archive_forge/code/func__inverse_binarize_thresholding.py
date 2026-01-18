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
def _inverse_binarize_thresholding(y, output_type, classes, threshold):
    """Inverse label binarization transformation using thresholding."""
    if output_type == 'binary' and y.ndim == 2 and (y.shape[1] > 2):
        raise ValueError("output_type='binary', but y.shape = {0}".format(y.shape))
    if output_type != 'binary' and y.shape[1] != len(classes):
        raise ValueError('The number of class is not equal to the number of dimension of y.')
    classes = np.asarray(classes)
    if sp.issparse(y):
        if threshold > 0:
            if y.format not in ('csr', 'csc'):
                y = y.tocsr()
            y.data = np.array(y.data > threshold, dtype=int)
            y.eliminate_zeros()
        else:
            y = np.array(y.toarray() > threshold, dtype=int)
    else:
        y = np.array(y > threshold, dtype=int)
    if output_type == 'binary':
        if sp.issparse(y):
            y = y.toarray()
        if y.ndim == 2 and y.shape[1] == 2:
            return classes[y[:, 1]]
        elif len(classes) == 1:
            return np.repeat(classes[0], len(y))
        else:
            return classes[y.ravel()]
    elif output_type == 'multilabel-indicator':
        return y
    else:
        raise ValueError('{0} format is not supported'.format(output_type))