import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
def _decade_less_equal(x, base):
    """
    Return the largest integer power of *base* that's less or equal to *x*.

    If *x* is negative, the exponent will be *greater*.
    """
    return x if x == 0 else -_decade_greater_equal(-x, base) if x < 0 else base ** np.floor(np.log(x) / np.log(base))