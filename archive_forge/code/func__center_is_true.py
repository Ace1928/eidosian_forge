import operator
import warnings
import numpy
import cupy
from cupy import _core
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
def _center_is_true(structure, origin):
    coor = tuple([oo + ss // 2 for ss, oo in zip(structure.shape, origin)])
    return bool(structure[coor])