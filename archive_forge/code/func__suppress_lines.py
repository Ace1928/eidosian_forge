import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, convolve
from ..transform import integral_image
from .corner import structure_tensor
from ..morphology import octagon, star
from .censure_cy import _censure_dob_loop
from ..feature.util import (
from .._shared.utils import check_nD
def _suppress_lines(feature_mask, image, sigma, line_threshold):
    Arr, Arc, Acc = structure_tensor(image, sigma, order='rc')
    feature_mask[(Arr + Acc) ** 2 > line_threshold * (Arr * Acc - Arc ** 2)] = False