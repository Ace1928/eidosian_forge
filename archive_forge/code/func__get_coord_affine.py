import numpy
import cupy
import cupy._core.internal
from cupyx.scipy.ndimage import _spline_prefilter_core
from cupyx.scipy.ndimage import _spline_kernel_weights
from cupyx.scipy.ndimage import _util
def _get_coord_affine(ndim, nprepad=0):
    """Compute target coordinate based on a homogeneous transformation matrix.

    The homogeneous matrix has shape (ndim, ndim + 1). It corresponds to
    affine matrix where the last row of the affine is assumed to be:
    ``[0] * ndim + [1]``.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        mat(array): array containing the (ndim, ndim + 1) transform matrix.
        in_coords(array): coordinates of the input

    For example, in 2D:

        c_0 = mat[0] * in_coords[0] + mat[1] * in_coords[1] + aff[2];
        c_1 = mat[3] * in_coords[0] + mat[4] * in_coords[1] + aff[5];

    """
    ops = []
    pre = f' + (W){nprepad}' if nprepad > 0 else ''
    ncol = ndim + 1
    for j in range(ndim):
        ops.append(f'\n            W c_{j} = (W)0.0;')
        for k in range(ndim):
            ops.append(f'\n            c_{j} += mat[{ncol * j + k}] * (W)in_coord[{k}];')
        ops.append(f'\n            c_{j} += mat[{ncol * j + ndim}]{pre};')
    return ops