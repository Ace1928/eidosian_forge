import warnings
from textwrap import dedent
import numpy as np
from . import _iterative
from scipy.sparse.linalg._interface import LinearOperator
from .utils import make_system
from scipy._lib._util import _aligned_zeros
from scipy._lib._threadsafety import non_reentrant
def _stoptest(residual, atol):
    """
    Successful termination condition for the solvers.
    """
    resid = np.linalg.norm(residual)
    if resid <= atol:
        return (resid, 1)
    else:
        return (resid, 0)