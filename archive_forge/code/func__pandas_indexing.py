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
def _pandas_indexing(X, key, key_dtype, axis):
    """Index a pandas dataframe or a series."""
    if _is_arraylike_not_scalar(key):
        key = np.asarray(key)
    if key_dtype == 'int' and (not (isinstance(key, slice) or np.isscalar(key))):
        return X.take(key, axis=axis)
    else:
        indexer = X.iloc if key_dtype == 'int' else X.loc
        return indexer[:, key] if axis else indexer[key]