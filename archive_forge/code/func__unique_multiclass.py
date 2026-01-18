import warnings
from collections.abc import Sequence
from itertools import chain
import numpy as np
from scipy.sparse import issparse
from ..utils._array_api import get_namespace
from ..utils.fixes import VisibleDeprecationWarning
from .validation import _assert_all_finite, check_array
def _unique_multiclass(y):
    xp, is_array_api_compliant = get_namespace(y)
    if hasattr(y, '__array__') or is_array_api_compliant:
        return xp.unique_values(xp.asarray(y))
    else:
        return set(y)