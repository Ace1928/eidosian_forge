import math
import numpy as np
import scipy.ndimage as ndi
from .._shared.utils import check_nD, _supported_float_type
from ..feature.util import DescriptorExtractor, FeatureDetector
from .._shared.filters import gaussian
from ..transform import rescale
from ..util import img_as_float
from ._sift import _local_max, _ori_distances, _update_histogram
def _set_number_of_octaves(self, image_shape):
    size_min = 12
    s0 = min(image_shape) * self.upsampling
    max_octaves = int(math.log2(s0 / size_min) + 1)
    if max_octaves < self.n_octaves:
        self.n_octaves = max_octaves