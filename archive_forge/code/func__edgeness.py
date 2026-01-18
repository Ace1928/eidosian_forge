import math
import numpy as np
import scipy.ndimage as ndi
from .._shared.utils import check_nD, _supported_float_type
from ..feature.util import DescriptorExtractor, FeatureDetector
from .._shared.filters import gaussian
from ..transform import rescale
from ..util import img_as_float
from ._sift import _local_max, _ori_distances, _update_histogram
def _edgeness(hxx, hyy, hxy):
    """Compute edgeness (eq. 18 of Otero et. al. IPOL paper)"""
    trace = hxx + hyy
    determinant = hxx * hyy - hxy * hxy
    return trace * trace / determinant