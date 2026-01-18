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
def _get_column_indices_interchange(X_interchange, key, key_dtype):
    """Same as _get_column_indices but for X with __dataframe__ protocol."""
    n_columns = X_interchange.num_columns()
    if isinstance(key, (list, tuple)) and (not key):
        return []
    elif key_dtype in ('bool', 'int'):
        return _get_column_indices_for_bool_or_int(key, n_columns)
    else:
        column_names = list(X_interchange.column_names())
        if isinstance(key, slice):
            if key.step not in [1, None]:
                raise NotImplementedError('key.step must be 1 or None')
            start, stop = (key.start, key.stop)
            if start is not None:
                start = column_names.index(start)
            if stop is not None:
                stop = column_names.index(stop) + 1
            else:
                stop = n_columns + 1
            return list(islice(range(n_columns), start, stop))
        selected_columns = [key] if np.isscalar(key) else key
        try:
            return [column_names.index(col) for col in selected_columns]
        except ValueError as e:
            raise ValueError('A given column is not a column of the dataframe') from e