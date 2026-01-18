import math
import numpy as np
import scipy.ndimage as ndi
from .._shared.utils import check_nD, _supported_float_type
from ..feature.util import DescriptorExtractor, FeatureDetector
from .._shared.filters import gaussian
from ..transform import rescale
from ..util import img_as_float
from ._sift import _local_max, _ori_distances, _update_histogram
def _inrange(self, a, dim):
    return (a[:, 0] > 0) & (a[:, 0] < dim[0] - 1) & (a[:, 1] > 0) & (a[:, 1] < dim[1] - 1)