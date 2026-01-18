import numpy
from .Qt import QtGui
from . import functions
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions
def _convert_2dlut_to_1dlut(xp, lut):
    augmented_alpha = False
    if lut.ndim == 1:
        return (lut, augmented_alpha)
    if lut.shape[1] == 3:
        lut = xp.column_stack([lut, xp.full(lut.shape[0], 255, dtype=xp.uint8)])
        augmented_alpha = True
    if lut.shape[1] == 4:
        lut = lut.view(xp.uint32)
    lut = lut.ravel()
    return (lut, augmented_alpha)