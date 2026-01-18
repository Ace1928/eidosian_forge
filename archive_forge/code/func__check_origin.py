import warnings
import numpy
import cupy
def _check_origin(origin, width):
    origin = int(origin)
    if width // 2 + origin < 0 or width // 2 + origin >= width:
        raise ValueError('invalid origin')
    return origin