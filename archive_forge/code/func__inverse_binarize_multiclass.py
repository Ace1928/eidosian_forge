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
def _inverse_binarize_multiclass(y, classes):
    """Inverse label binarization transformation for multiclass.

    Multiclass uses the maximal score instead of a threshold.
    """
    classes = np.asarray(classes)
    if sp.issparse(y):
        y = y.tocsr()
        n_samples, n_outputs = y.shape
        outputs = np.arange(n_outputs)
        row_max = min_max_axis(y, 1)[1]
        row_nnz = np.diff(y.indptr)
        y_data_repeated_max = np.repeat(row_max, row_nnz)
        y_i_all_argmax = np.flatnonzero(y_data_repeated_max == y.data)
        if row_max[-1] == 0:
            y_i_all_argmax = np.append(y_i_all_argmax, [len(y.data)])
        index_first_argmax = np.searchsorted(y_i_all_argmax, y.indptr[:-1])
        y_ind_ext = np.append(y.indices, [0])
        y_i_argmax = y_ind_ext[y_i_all_argmax[index_first_argmax]]
        y_i_argmax[np.where(row_nnz == 0)[0]] = 0
        samples = np.arange(n_samples)[(row_nnz > 0) & (row_max.ravel() == 0)]
        for i in samples:
            ind = y.indices[y.indptr[i]:y.indptr[i + 1]]
            y_i_argmax[i] = classes[np.setdiff1d(outputs, ind)][0]
        return classes[y_i_argmax]
    else:
        return classes.take(y.argmax(axis=1), mode='clip')