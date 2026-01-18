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
def _get_column_indices_for_bool_or_int(key, n_columns):
    try:
        idx = _safe_indexing(np.arange(n_columns), key)
    except IndexError as e:
        raise ValueError(f'all features must be in [0, {n_columns - 1}] or [-{n_columns}, 0]') from e
    return np.atleast_1d(idx).tolist()