import numpy as np
from ..feature.util import (
from .corner import corner_fast, corner_orientations, corner_peaks, corner_harris
from ..transform import pyramid_gaussian
from .._shared.utils import check_nD
from .._shared.compat import NP_COPY_IF_NEEDED
from .orb_cy import _orb_loop
def _build_pyramid(self, image):
    image = _prepare_grayscale_input_2D(image)
    return list(pyramid_gaussian(image, self.n_scales - 1, self.downscale, channel_axis=None))