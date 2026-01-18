import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy.signal import convolve2d, convolve
from skimage import restoration, util
from skimage._shared import filters
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.data import astronaut, camera
from skimage.restoration import uft
def _get_rtol_atol(dtype):
    rtol = 0.001
    atol = 0
    if dtype == np.float16:
        rtol = 0.01
        atol = 0.001
    elif dtype == np.float32:
        atol = 1e-05
    return (rtol, atol)