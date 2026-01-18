import numpy as np
from scipy import ndimage as ndi
from ._geometric import SimilarityTransform, AffineTransform, ProjectiveTransform
from ._warps_cy import _warp_fast
from ..measure import block_reduce
from .._shared.utils import (
def _log_polar_mapping(output_coords, k_angle, k_radius, center):
    """Inverse mapping function to convert from cartesian to polar coordinates

    Parameters
    ----------
    output_coords : (M, 2) ndarray
        Array of `(col, row)` coordinates in the output image.
    k_angle : float
        Scaling factor that relates the intended number of rows in the output
        image to angle: ``k_angle = nrows / (2 * np.pi)``.
    k_radius : float
        Scaling factor that relates the radius of the circle bounding the
        area to be transformed to the intended number of columns in the output
        image: ``k_radius = width / np.log(radius)``.
    center : 2-tuple
        `(row, col)` coordinates that represent the center of the circle that bounds the
        area to be transformed in an input image.

    Returns
    -------
    coords : ndarray, shape (M, 2)
        Array of `(col, row)` coordinates in the input image that
        correspond to the `output_coords` given as input.
    """
    angle = output_coords[:, 1] / k_angle
    rr = np.exp(output_coords[:, 0] / k_radius) * np.sin(angle) + center[0]
    cc = np.exp(output_coords[:, 0] / k_radius) * np.cos(angle) + center[1]
    coords = np.column_stack((cc, rr))
    return coords