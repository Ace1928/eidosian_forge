import math
import warnings
import numpy as np
import dataclasses
from typing import Optional, Callable
from functools import partial
from scipy._lib._util import _asarray_validated
from . import _distance_wrap
from . import _hausdorff
from ..linalg import norm
from ..special import rel_entr
from . import _distance_pybind
def _validate_minkowski_kwargs(X, m, n, **kwargs):
    kwargs = _validate_weight_with_size(X, m, n, **kwargs)
    if 'p' not in kwargs:
        kwargs['p'] = 2.0
    elif kwargs['p'] <= 0:
        raise ValueError('p must be greater than 0')
    return kwargs