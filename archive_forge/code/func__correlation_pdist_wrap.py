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
def _correlation_pdist_wrap(X, dm, **kwargs):
    X2 = X - X.mean(axis=1, keepdims=True)
    _distance_wrap.pdist_cosine_double_wrap(X2, dm, **kwargs)