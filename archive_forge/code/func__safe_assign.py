import math
import numbers
import platform
import struct
import timeit
import warnings
from collections.abc import Sequence
from contextlib import contextmanager, suppress
from itertools import compress, islice
import numpy as np
from scipy.sparse import issparse
from .. import get_config
from ..exceptions import DataConversionWarning
from . import _joblib, metadata_routing
from ._bunch import Bunch
from ._estimator_html_repr import estimator_html_repr
from ._param_validation import Integral, Interval, validate_params
from .class_weight import compute_class_weight, compute_sample_weight
from .deprecation import deprecated
from .discovery import all_estimators
from .fixes import parse_version, threadpool_info
from .murmurhash import murmurhash3_32
from .validation import (
def _safe_assign(X, values, *, row_indexer=None, column_indexer=None):
    """Safe assignment to a numpy array, sparse matrix, or pandas dataframe.

    Parameters
    ----------
    X : {ndarray, sparse-matrix, dataframe}
        Array to be modified. It is expected to be 2-dimensional.

    values : ndarray
        The values to be assigned to `X`.

    row_indexer : array-like, dtype={int, bool}, default=None
        A 1-dimensional array to select the rows of interest. If `None`, all
        rows are selected.

    column_indexer : array-like, dtype={int, bool}, default=None
        A 1-dimensional array to select the columns of interest. If `None`, all
        columns are selected.
    """
    row_indexer = slice(None, None, None) if row_indexer is None else row_indexer
    column_indexer = slice(None, None, None) if column_indexer is None else column_indexer
    if hasattr(X, 'iloc'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            X.iloc[row_indexer, column_indexer] = values
    else:
        X[row_indexer, column_indexer] = values