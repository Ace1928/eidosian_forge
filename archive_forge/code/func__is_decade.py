import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
def _is_decade(x, *, base=10, rtol=None):
    """Return True if *x* is an integer power of *base*."""
    if not np.isfinite(x):
        return False
    if x == 0.0:
        return True
    lx = np.log(abs(x)) / np.log(base)
    if rtol is None:
        return np.isclose(lx, np.round(lx))
    else:
        return np.isclose(lx, np.round(lx), rtol=rtol)