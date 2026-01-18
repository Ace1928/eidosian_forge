import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def _atleast_2d(*arys):
    """
    Version of `np.atleast_2d`, copied from
    https://github.com/numpy/numpy/blob/master/numpy/core/shape_base.py,
    with the following modifications:

    1. It allows for `None` arguments, and passes them directly through
    2. Instead of creating new axis at the beginning, it creates it at the end
    """
    res = []
    for ary in arys:
        if ary is None:
            result = None
        else:
            ary = np.asanyarray(ary)
            if ary.ndim == 0:
                result = ary.reshape(1, 1)
            elif ary.ndim == 1:
                result = ary[:, np.newaxis]
            else:
                result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res