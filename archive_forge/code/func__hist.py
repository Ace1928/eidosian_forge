import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util
def _hist(vals):
    return cupy.histogram(vals, _bins)[0]