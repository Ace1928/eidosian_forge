import collections
from collections import OrderedDict
import re
import warnings
from contextlib import contextmanager
from copy import deepcopy, copy
import itertools
from functools import reduce
from _plotly_utils.utils import (
from _plotly_utils.exceptions import PlotlyKeyError
from .optional_imports import get_module
from . import shapeannotation
from . import _subplots
@staticmethod
def _vals_equal(v1, v2):
    """
        Recursive equality function that handles nested dicts / tuples / lists
        that contain numpy arrays.

        v1
            First value to compare
        v2
            Second value to compare

        Returns
        -------
        bool
            True if v1 and v2 are equal, False otherwise
        """
    np = get_module('numpy', should_load=False)
    if np is not None and (isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray)):
        return np.array_equal(v1, v2)
    elif isinstance(v1, (list, tuple)):
        return isinstance(v2, (list, tuple)) and len(v1) == len(v2) and all((BasePlotlyType._vals_equal(e1, e2) for e1, e2 in zip(v1, v2)))
    elif isinstance(v1, dict):
        return isinstance(v2, dict) and set(v1.keys()) == set(v2.keys()) and all((BasePlotlyType._vals_equal(v1[k], v2[k]) for k in v1))
    else:
        return v1 == v2