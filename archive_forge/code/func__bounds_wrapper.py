from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def _bounds_wrapper(t, y, p=(), be=None):
    if lb is not None:
        if np.any(y < lb - 10 * self._current_integration_kwargs['atol']):
            raise RecoverableError
        y = np.array(y)
        y[y < lb] = lb[y < lb]
    if ub is not None:
        if np.any(y > ub + 10 * self._current_integration_kwargs['atol']):
            raise RecoverableError
        y = np.array(y)
        y[y > ub] = ub[y > ub]
    return cb(t, y, p, be)