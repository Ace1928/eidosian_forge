import functools
import importlib
import importlib.resources
import re
import warnings
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis
from .rcparams import rcParams
def _numba_var(numba_function, standard_numpy_func, data, axis=None, ddof=0):
    """Replace the numpy methods used to calculate variance.

    Parameters
    ----------
    numba_function : function()
        Custom numba function included in stats/stats_utils.py.

    standard_numpy_func: function()
        Standard function included in the numpy library.

    data : array.
    axis : axis along which the variance is calculated.
    ddof : degrees of freedom allowed while calculating variance.

    Returns
    -------
    array:
        variance values calculate by appropriate function for numba speedup
        if Numba is installed or enabled.

    """
    if Numba.numba_flag:
        return numba_function(data, axis=axis, ddof=ddof)
    else:
        return standard_numpy_func(data, axis=axis, ddof=ddof)