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
def _validate_mahalanobis_kwargs(X, m, n, **kwargs):
    VI = kwargs.pop('VI', None)
    if VI is None:
        if m <= n:
            raise ValueError('The number of observations (%d) is too small; the covariance matrix is singular. For observations with %d dimensions, at least %d observations are required.' % (m, n, n + 1))
        if isinstance(X, tuple):
            X = np.vstack(X)
        CV = np.atleast_2d(np.cov(X.astype(np.float64, copy=False).T))
        VI = np.linalg.inv(CV).T.copy()
    kwargs['VI'] = _convert_to_double(VI)
    return kwargs