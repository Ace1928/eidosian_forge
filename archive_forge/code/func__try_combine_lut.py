import numpy
from .Qt import QtGui
from . import functions
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions
def _try_combine_lut(xp, image, levels, lut):
    augmented_alpha = False
    if image.dtype == xp.uint16 and levels is None and (image.ndim == 3) and (image.shape[2] == 3):
        levels = [0, 65535]
    if levels is None and lut is None:
        return (image, levels, lut, augmented_alpha)
    levels_lut = None
    colors_lut = lut
    eflsize = 2 ** (image.itemsize * 8)
    if levels is None:
        info = xp.iinfo(image.dtype)
        minlev, maxlev = (info.min, info.max)
    else:
        minlev, maxlev = levels
    levdiff = maxlev - minlev
    levdiff = 1 if levdiff == 0 else levdiff
    if colors_lut is None:
        if image.dtype == xp.ubyte and image.ndim == 2:
            ind = xp.arange(eflsize)
            levels_lut = functions.rescaleData(ind, scale=255.0 / levdiff, offset=minlev, dtype=xp.ubyte)
            return (image, None, levels_lut, augmented_alpha)
        else:
            image = functions.rescaleData(image, scale=255.0 / levdiff, offset=minlev, dtype=xp.ubyte)
            return (image, None, colors_lut, augmented_alpha)
    else:
        num_colors = colors_lut.shape[0]
        effscale = num_colors / levdiff
        lutdtype = xp.min_scalar_type(num_colors - 1)
        if image.dtype == xp.ubyte or lutdtype != xp.ubyte:
            ind = xp.arange(eflsize)
            levels_lut = functions.rescaleData(ind, scale=effscale, offset=minlev, dtype=lutdtype, clip=(0, num_colors - 1))
            efflut = colors_lut[levels_lut]
            if image.dtype == xp.uint16 and image.ndim == 2:
                image, augmented_alpha = _apply_lut_for_uint16_mono(xp, image, efflut)
                efflut = None
            return (image, None, efflut, augmented_alpha)
        else:
            image = functions.rescaleData(image, scale=effscale, offset=minlev, dtype=lutdtype, clip=(0, num_colors - 1))
            return (image, None, colors_lut, augmented_alpha)