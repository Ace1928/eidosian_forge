import numbers
import operator
import sys
import warnings
from contextlib import suppress
from functools import reduce, wraps
from inspect import Parameter, isclass, signature
import joblib
import numpy as np
import scipy.sparse as sp
from .. import get_config as _get_config
from ..exceptions import DataConversionWarning, NotFittedError, PositiveSpectrumWarning
from ..utils._array_api import _asarray_with_order, _is_numpy_namespace, get_namespace
from ..utils.fixes import ComplexWarning, _preserve_dia_indices_dtype
from ._isfinite import FiniteStatus, cy_isfinite
from .fixes import _object_dtype_isnan
def _check_pos_label_consistency(pos_label, y_true):
    """Check if `pos_label` need to be specified or not.

    In binary classification, we fix `pos_label=1` if the labels are in the set
    {-1, 1} or {0, 1}. Otherwise, we raise an error asking to specify the
    `pos_label` parameters.

    Parameters
    ----------
    pos_label : int, float, bool, str or None
        The positive label.
    y_true : ndarray of shape (n_samples,)
        The target vector.

    Returns
    -------
    pos_label : int, float, bool or str
        If `pos_label` can be inferred, it will be returned.

    Raises
    ------
    ValueError
        In the case that `y_true` does not have label in {-1, 1} or {0, 1},
        it will raise a `ValueError`.
    """
    classes = np.unique(y_true)
    if pos_label is None and (classes.dtype.kind in 'OUS' or not (np.array_equal(classes, [0, 1]) or np.array_equal(classes, [-1, 1]) or np.array_equal(classes, [0]) or np.array_equal(classes, [-1]) or np.array_equal(classes, [1]))):
        classes_repr = ', '.join([repr(c) for c in classes.tolist()])
        raise ValueError(f'y_true takes value in {{{classes_repr}}} and pos_label is not specified: either make y_true take value in {{0, 1}} or {{-1, 1}} or pass pos_label explicitly.')
    elif pos_label is None:
        pos_label = 1
    return pos_label