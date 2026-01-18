import numpy
from .Qt import QtGui
from . import functions
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions
def _rescale_float_mono(xp, image, levels, lut):
    augmented_alpha = False
    if lut is not None:
        scale = lut.shape[0]
        num_colors = lut.shape[0]
    else:
        scale = 255.0
        num_colors = 256
    dtype = xp.min_scalar_type(num_colors - 1)
    minVal, maxVal = levels
    rng = maxVal - minVal
    rng = 1 if rng == 0 else rng
    fn_numba = getNumbaFunctions()
    if xp == numpy and image.flags.c_contiguous and (dtype == xp.uint16) and (fn_numba is not None):
        lut, augmented_alpha = _convert_2dlut_to_1dlut(xp, lut)
        image = fn_numba.rescale_and_lookup1d(image, scale / rng, minVal, lut)
        if image.dtype == xp.uint32:
            image = image[..., xp.newaxis].view(xp.uint8)
        return (image, None, None, augmented_alpha)
    else:
        image = functions.rescaleData(image, scale / rng, offset=minVal, dtype=dtype, clip=(0, num_colors - 1))
        levels = None
        if image.dtype == xp.uint16 and image.ndim == 2:
            image, augmented_alpha = _apply_lut_for_uint16_mono(xp, image, lut)
            lut = None
        return (image, levels, lut, augmented_alpha)