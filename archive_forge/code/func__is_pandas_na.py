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
def _is_pandas_na(x):
    """Test if x is pandas.NA.

    We intentionally do not use this function to return `True` for `pd.NA` in
    `is_scalar_nan`, because estimators that support `pd.NA` are the exception
    rather than the rule at the moment. When `pd.NA` is more universally
    supported, we may reconsider this decision.

    Parameters
    ----------
    x : any type

    Returns
    -------
    boolean
    """
    with suppress(ImportError):
        from pandas import NA
        return x is NA
    return False