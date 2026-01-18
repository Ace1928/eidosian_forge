import numpy as np
import scipy.optimize
from .utils import MAX_MEM_BLOCK
from typing import Any, Optional, Tuple, Sequence
def _nnls_lbfgs_block(A: np.ndarray, B: np.ndarray, x_init: Optional[np.ndarray]=None, **kwargs: Any) -> np.ndarray:
    """Solve the constrained problem over a single block

    Parameters
    ----------
    A : np.ndarray [shape=(m, d)]
        The basis matrix
    B : np.ndarray [shape=(m, N)]
        The regression targets
    x_init : np.ndarray [shape=(d, N)]
        An initial guess
    **kwargs
        Additional keyword arguments to `scipy.optimize.fmin_l_bfgs_b`

    Returns
    -------
    x : np.ndarray [shape=(d, N)]
        Non-negative matrix such that Ax ~= B
    """
    if x_init is None:
        x_init = np.einsum('fm,...mt->...ft', np.linalg.pinv(A), B, optimize=True)
        np.clip(x_init, 0, None, out=x_init)
    kwargs.setdefault('m', A.shape[1])
    bounds = [(0, None)] * x_init.size
    shape = x_init.shape
    x: np.ndarray
    x, obj_value, diagnostics = scipy.optimize.fmin_l_bfgs_b(_nnls_obj, x_init, args=(shape, A, B), bounds=bounds, **kwargs)
    return x.reshape(shape)