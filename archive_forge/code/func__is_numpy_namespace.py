import itertools
import math
from functools import wraps
import numpy
import scipy.special as special
from .._config import get_config
from .fixes import parse_version
def _is_numpy_namespace(xp):
    """Return True if xp is backed by NumPy."""
    return xp.__name__ in {'numpy', 'array_api_compat.numpy', 'numpy.array_api'}