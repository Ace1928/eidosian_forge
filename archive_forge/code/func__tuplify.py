from statsmodels.compat.python import lrange, lzip
from itertools import product
import numpy as np
from numpy import array, cumsum, iterable, r_
from pandas import DataFrame
from statsmodels.graphics import utils
def _tuplify(obj):
    """convert an object in a tuple of strings (even if it is not iterable,
    like a single integer number, but keep the string healthy)
    """
    if np.iterable(obj) and (not isinstance(obj, str)):
        res = tuple((str(o) for o in obj))
    else:
        res = (str(obj),)
    return res