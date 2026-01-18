import numpy as np
import numpy.linalg as L
from scipy.linalg import solveh_banded
from scipy.optimize import golden
from models import _hbspline     #removed because this was segfaulting
import warnings
def _upper2lower(ub):
    """
    Convert upper triangular banded matrix to lower banded form.

    INPUTS:
       ub  -- an upper triangular banded matrix

    OUTPUTS: lb
       lb  -- a lower triangular banded matrix with same entries
              as ub
    """
    lb = np.zeros(ub.shape, ub.dtype)
    nrow, ncol = ub.shape
    for i in range(ub.shape[0]):
        lb[i, 0:ncol - i] = ub[nrow - 1 - i, i:ncol]
        lb[i, ncol - i:] = ub[nrow - 1 - i, 0:i]
    return lb