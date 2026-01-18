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
def _check_large_sparse(X, accept_large_sparse=False):
    """Raise a ValueError if X has 64bit indices and accept_large_sparse=False"""
    if not accept_large_sparse:
        supported_indices = ['int32']
        if X.format == 'coo':
            index_keys = ['col', 'row']
        elif X.format in ['csr', 'csc', 'bsr']:
            index_keys = ['indices', 'indptr']
        else:
            return
        for key in index_keys:
            indices_datatype = getattr(X, key).dtype
            if indices_datatype not in supported_indices:
                raise ValueError(f'Only sparse matrices with 32-bit integer indices are accepted. Got {indices_datatype} indices. Please do report a minimal reproducer on scikit-learn issue tracker so that support for your use-case can be studied by maintainers. See: https://scikit-learn.org/dev/developers/minimal_reproducer.html')