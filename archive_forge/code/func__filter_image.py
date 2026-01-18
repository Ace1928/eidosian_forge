import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, convolve
from ..transform import integral_image
from .corner import structure_tensor
from ..morphology import octagon, star
from .censure_cy import _censure_dob_loop
from ..feature.util import (
from .._shared.utils import check_nD
def _filter_image(image, min_scale, max_scale, mode):
    response = np.zeros((image.shape[0], image.shape[1], max_scale - min_scale + 1), dtype=np.float64)
    if mode == 'dob':
        item_size = response.itemsize
        response.strides = (item_size * response.shape[1], item_size, item_size * response.shape[0] * response.shape[1])
        integral_img = integral_image(image)
        for i in range(max_scale - min_scale + 1):
            n = min_scale + i
            inner_weight = 1.0 / (2 * n + 1) ** 2
            outer_weight = 1.0 / (12 * n ** 2 + 4 * n)
            _censure_dob_loop(n, integral_img, response[:, :, i], inner_weight, outer_weight)
    elif mode == 'octagon':
        for i in range(max_scale - min_scale + 1):
            mo, no = OCTAGON_OUTER_SHAPE[min_scale + i - 1]
            mi, ni = OCTAGON_INNER_SHAPE[min_scale + i - 1]
            response[:, :, i] = convolve(image, _octagon_kernel(mo, no, mi, ni))
    elif mode == 'star':
        for i in range(max_scale - min_scale + 1):
            m = STAR_SHAPE[STAR_FILTER_SHAPE[min_scale + i - 1][0]]
            n = STAR_SHAPE[STAR_FILTER_SHAPE[min_scale + i - 1][1]]
            response[:, :, i] = convolve(image, _star_kernel(m, n))
    return response