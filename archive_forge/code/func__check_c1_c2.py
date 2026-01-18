from warnings import warn
from scipy.optimize import _minpack2 as minpack2    # noqa: F401
from ._dcsrch import DCSRCH
import numpy as np
def _check_c1_c2(c1, c2):
    if not 0 < c1 < c2 < 1:
        raise ValueError("'c1' and 'c2' do not satisfy'0 < c1 < c2 < 1'.")