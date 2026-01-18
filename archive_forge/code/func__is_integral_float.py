import warnings
from collections.abc import Sequence
from itertools import chain
import numpy as np
from scipy.sparse import issparse
from ..utils._array_api import get_namespace
from ..utils.fixes import VisibleDeprecationWarning
from .validation import _assert_all_finite, check_array
def _is_integral_float(y):
    xp, is_array_api_compliant = get_namespace(y)
    return xp.isdtype(y.dtype, 'real floating') and bool(xp.all(xp.astype(xp.astype(y, xp.int64), y.dtype) == y))