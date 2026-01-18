import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import weakref
import numpy as np
import matplotlib
from matplotlib import _api, _c_internal_utils
def _safe_first_finite(obj, *, skip_nonfinite=True):
    """
    Return the first finite element in *obj* if one is available and skip_nonfinite is
    True. Otherwise, return the first element.

    This is a method for internal use.

    This is a type-independent way of obtaining the first finite element, supporting
    both index access and the iterator protocol.
    """

    def safe_isfinite(val):
        if val is None:
            return False
        try:
            return math.isfinite(val)
        except (TypeError, ValueError):
            pass
        try:
            return np.isfinite(val) if np.isscalar(val) else True
        except TypeError:
            return True
    if skip_nonfinite is False:
        if isinstance(obj, collections.abc.Iterator):
            try:
                return obj[0]
            except TypeError:
                pass
            raise RuntimeError('matplotlib does not support generators as input')
        return next(iter(obj))
    elif isinstance(obj, np.flatiter):
        return obj[0]
    elif isinstance(obj, collections.abc.Iterator):
        raise RuntimeError('matplotlib does not support generators as input')
    else:
        for val in obj:
            if safe_isfinite(val):
                return val
        return safe_first_element(obj)