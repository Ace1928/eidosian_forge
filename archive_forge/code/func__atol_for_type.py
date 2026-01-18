import itertools
import math
from functools import wraps
import numpy
import scipy.special as special
from .._config import get_config
from .fixes import parse_version
def _atol_for_type(dtype):
    """Return the absolute tolerance for a given dtype."""
    return numpy.finfo(dtype).eps * 100