import numpy as np
from scipy import ndimage as ndi
from ..transform import pyramid_reduce
from ..util.dtype import _convert
def _get_warp_points(grid, flow):
    """Compute warp point coordinates.

    Parameters
    ----------
    grid : iterable
        The sparse grid to be warped (obtained using
        ``np.meshgrid(..., sparse=True)).``)
    flow : ndarray
        The warping motion field.

    Returns
    -------
    out : ndarray
        The warp point coordinates.

    """
    out = flow.copy()
    for idx, g in enumerate(grid):
        out[idx, ...] += g
    return out