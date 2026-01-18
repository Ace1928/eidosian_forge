from warnings import warn
from itertools import product
import numpy as np
from numpy import atleast_1d, atleast_2d
from .lapack import get_lapack_funcs, _compute_lwork
from ._misc import LinAlgError, _datacopied, LinAlgWarning
from ._decomp import _asarray_validated
from . import _decomp, _decomp_svd
from ._solve_toeplitz import levinson
from ._cythonized_array_utils import find_det_from_lu
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
from scipy.linalg._flinalg_py import get_flinalg_funcs  # noqa: F401
def _solve_check(n, info, lamch=None, rcond=None):
    """ Check arguments during the different steps of the solution phase """
    if info < 0:
        raise ValueError(f'LAPACK reported an illegal value in {-info}-th argument.')
    elif 0 < info:
        raise LinAlgError('Matrix is singular.')
    if lamch is None:
        return
    E = lamch('E')
    if rcond < E:
        warn(f'Ill-conditioned matrix (rcond={rcond:.6g}): result may not be accurate.', LinAlgWarning, stacklevel=3)