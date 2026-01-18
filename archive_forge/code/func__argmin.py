import re
from contextlib import contextmanager
import functools
import operator
import warnings
import numbers
from collections import namedtuple
import inspect
import math
from typing import (
import numpy as np
from scipy._lib._array_api import array_namespace
def _argmin(a, keepdims=False, axis=None):
    """
    argmin with a `keepdims` parameter.

    See https://github.com/numpy/numpy/issues/8710

    If axis is not None, a.shape[axis] must be greater than 0.
    """
    res = np.argmin(a, axis=axis)
    if keepdims and axis is not None:
        res = np.expand_dims(res, axis=axis)
    return res