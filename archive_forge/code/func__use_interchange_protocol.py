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
def _use_interchange_protocol(X):
    """Use interchange protocol for non-pandas dataframes that follow the protocol.

    Note: at this point we chose not to use the interchange API on pandas dataframe
    to ensure strict behavioral backward compatibility with older versions of
    scikit-learn.
    """
    return not _is_pandas_df(X) and hasattr(X, '__dataframe__')