import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, convolve
from ..transform import integral_image
from .corner import structure_tensor
from ..morphology import octagon, star
from .censure_cy import _censure_dob_loop
from ..feature.util import (
from .._shared.utils import check_nD
def _star_kernel(m, n):
    c = m + m // 2 - n - n // 2
    outer_star = star(m)
    inner_star = np.zeros_like(outer_star)
    inner_star[c:-c, c:-c] = star(n)
    outer_weight = 1.0 / np.sum(outer_star - inner_star)
    inner_weight = 1.0 / np.sum(inner_star)
    bfilter = outer_weight * outer_star - (outer_weight + inner_weight) * inner_star
    return bfilter