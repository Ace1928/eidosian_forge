import math
import numpy as np
import scipy.ndimage as ndi
from .._shared.utils import check_nD, _supported_float_type
from ..feature.util import DescriptorExtractor, FeatureDetector
from .._shared.filters import gaussian
from ..transform import rescale
from ..util import img_as_float
from ._sift import _local_max, _ori_distances, _update_histogram
def _sparse_gradient(vol, positions):
    """Gradient of a 3D volume at the provided `positions`.

    For SIFT we only need the gradient at specific positions and do not need
    the gradient at the edge positions, so can just use this simple
    implementation instead of numpy.gradient.
    """
    p0 = positions[..., 0]
    p1 = positions[..., 1]
    p2 = positions[..., 2]
    g0 = vol[p0 + 1, p1, p2] - vol[p0 - 1, p1, p2]
    g0 *= 0.5
    g1 = vol[p0, p1 + 1, p2] - vol[p0, p1 - 1, p2]
    g1 *= 0.5
    g2 = vol[p0, p1, p2 + 1] - vol[p0, p1, p2 - 1]
    g2 *= 0.5
    return (g0, g1, g2)