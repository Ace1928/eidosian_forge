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
def _copy_array_if_base_present(a):
    """Copy the array if its base points to a parent array."""
    if a.base is not None:
        return a.copy()
    return a