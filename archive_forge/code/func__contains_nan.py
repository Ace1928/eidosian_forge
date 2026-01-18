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
def _contains_nan(a, nan_policy='propagate', use_summation=True, policies=None):
    if not isinstance(a, np.ndarray):
        use_summation = False
    if policies is None:
        policies = ['propagate', 'raise', 'omit']
    if nan_policy not in policies:
        raise ValueError('nan_policy must be one of {%s}' % ', '.join(("'%s'" % s for s in policies)))
    if np.issubdtype(a.dtype, np.inexact):
        if use_summation:
            with np.errstate(invalid='ignore', over='ignore'):
                contains_nan = np.isnan(np.sum(a))
        else:
            contains_nan = np.isnan(a).any()
    elif np.issubdtype(a.dtype, object):
        contains_nan = False
        for el in a.ravel():
            if np.issubdtype(type(el), np.number) and np.isnan(el):
                contains_nan = True
                break
    else:
        contains_nan = False
    if contains_nan and nan_policy == 'raise':
        raise ValueError('The input contains nan values')
    return (contains_nan, nan_policy)