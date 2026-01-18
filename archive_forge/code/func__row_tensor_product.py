import numpy as np
from patsy.util import (have_pandas, atleast_2d_column_default,
from patsy.state import stateful_transform
def _row_tensor_product(dms):
    """Computes row-wise tensor product of given arguments.

    .. note:: Custom algorithm to precisely match what is done in 'mgcv',
    in particular look out for order of result columns!
    For reference implementation see 'mgcv' source code,
    file 'mat.c', mgcv_tensor_mm(), l.62

    :param dms: A sequence of 2-d arrays (marginal design matrices).
    :return: The 2-d array row-wise tensor product of given arguments.

    :raise ValueError: if argument sequence is empty, does not contain only
     2-d arrays or if the arrays number of rows does not match.
    """
    if len(dms) == 0:
        raise ValueError('Tensor product arrays sequence should not be empty.')
    for dm in dms:
        if dm.ndim != 2:
            raise ValueError('Tensor product arguments should be 2-d arrays.')
    tp_nrows = dms[0].shape[0]
    tp_ncols = 1
    for dm in dms:
        if dm.shape[0] != tp_nrows:
            raise ValueError('Tensor product arguments should have same number of rows.')
        tp_ncols *= dm.shape[1]
    tp = np.zeros((tp_nrows, tp_ncols))
    tp[:, -dms[-1].shape[1]:] = dms[-1]
    filled_tp_ncols = dms[-1].shape[1]
    for dm in dms[-2::-1]:
        p = -filled_tp_ncols * dm.shape[1]
        for j in range(dm.shape[1]):
            xj = dm[:, j]
            for t in range(-filled_tp_ncols, 0):
                tp[:, p] = tp[:, t] * xj
                p += 1
        filled_tp_ncols *= dm.shape[1]
    return tp