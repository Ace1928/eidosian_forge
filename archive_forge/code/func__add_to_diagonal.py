import itertools
import math
from functools import wraps
import numpy
import scipy.special as special
from .._config import get_config
from .fixes import parse_version
def _add_to_diagonal(array, value, xp):
    value = xp.asarray(value, dtype=array.dtype)
    if _is_numpy_namespace(xp):
        array_np = numpy.asarray(array)
        array_np.flat[::array.shape[0] + 1] += value
        return xp.asarray(array_np)
    elif value.ndim == 1:
        for i in range(array.shape[0]):
            array[i, i] += value[i]
    else:
        for i in range(array.shape[0]):
            array[i, i] += value