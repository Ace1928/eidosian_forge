import numpy as np
from ..util.dtype import dtype_range, dtype_limits
from .._shared import utils
def _adjust_gamma_u8(image, gamma, gain):
    """LUT based implementation of gamma adjustment."""
    lut = 255 * gain * np.linspace(0, 1, 256) ** gamma
    lut = np.minimum(np.rint(lut), 255).astype('uint8')
    return lut[image]