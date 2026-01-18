import numpy as np
from scipy import ndimage as ndi
from ._geometric import SimilarityTransform, AffineTransform, ProjectiveTransform
from ._warps_cy import _warp_fast
from ..measure import block_reduce
from .._shared.utils import (
def _swirl_mapping(xy, center, rotation, strength, radius):
    x, y = xy.T
    x0, y0 = center
    rho = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    radius = radius / 5 * np.log(2)
    theta = rotation + strength * np.exp(-rho / radius) + np.arctan2(y - y0, x - x0)
    xy[..., 0] = x0 + rho * np.cos(theta)
    xy[..., 1] = y0 + rho * np.sin(theta)
    return xy