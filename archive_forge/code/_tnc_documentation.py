from scipy.optimize import _moduleTNC as moduleTNC
from ._optimize import (MemoizeJac, OptimizeResult, _check_unknown_options,
from ._constraints import old_bound_to_new
from scipy._lib._array_api import atleast_nd, array_namespace
from numpy import inf, array, zeros

    low, up   : the bounds (lists of floats)
                if low is None, the lower bounds are removed.
                if up is None, the upper bounds are removed.
                low and up defaults to None
    