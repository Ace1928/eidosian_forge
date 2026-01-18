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
def _allclose_dense_sparse(x, y, rtol=1e-07, atol=1e-09):
    """Check allclose for sparse and dense data.

    Both x and y need to be either sparse or dense, they
    can't be mixed.

    Parameters
    ----------
    x : {array-like, sparse matrix}
        First array to compare.

    y : {array-like, sparse matrix}
        Second array to compare.

    rtol : float, default=1e-7
        Relative tolerance; see numpy.allclose.

    atol : float, default=1e-9
        absolute tolerance; see numpy.allclose. Note that the default here is
        more tolerant than the default for numpy.testing.assert_allclose, where
        atol=0.
    """
    if sp.issparse(x) and sp.issparse(y):
        x = x.tocsr()
        y = y.tocsr()
        x.sum_duplicates()
        y.sum_duplicates()
        return np.array_equal(x.indices, y.indices) and np.array_equal(x.indptr, y.indptr) and np.allclose(x.data, y.data, rtol=rtol, atol=atol)
    elif not sp.issparse(x) and (not sp.issparse(y)):
        return np.allclose(x, y, rtol=rtol, atol=atol)
    raise ValueError('Can only compare two sparse matrices, not a sparse matrix and an array')