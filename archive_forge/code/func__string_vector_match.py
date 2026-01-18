from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import re
import sys
import warnings
def _string_vector_match(data, match, fun, dtype=str):
    """Get a boolean match array from a vector.

    Parameters
    ----------
    data : list-like
        Vector to be matched against
    match : `dtype` or list-like
        Match criteria
    fun : callable(x, match)
        Function that returns True if `match` matches `x`
    dtype : type, optional (default: str)
        Expected type(match) (if not list-like)

    Returns
    -------
    data_match : list-like, dtype=bool
    """
    if isinstance(match, dtype):
        fun = np.vectorize(fun)
        return fun(data, match)
    else:
        return np.any([_string_vector_match(data, m, fun, dtype=dtype) for m in match], axis=0)