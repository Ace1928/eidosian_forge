import operator
import warnings
import numpy
import cupy
from cupy import _core
from cupy._core import _accelerator
from cupy.cuda import cub
from cupy.cuda import common
from cupy.cuda import runtime
def _get_outer_edges(a, range):
    """
    Determine the outer bin edges to use, from either the data or the range
    argument
    """
    if range is not None:
        first_edge, last_edge = range
        if first_edge > last_edge:
            raise ValueError('max must be larger than min in range parameter.')
        if not (numpy.isfinite(first_edge) and numpy.isfinite(last_edge)):
            raise ValueError('supplied range of [{}, {}] is not finite'.format(first_edge, last_edge))
    elif a.size == 0:
        first_edge = 0.0
        last_edge = 1.0
    else:
        first_edge = float(a.min())
        last_edge = float(a.max())
        if not (cupy.isfinite(first_edge) and cupy.isfinite(last_edge)):
            raise ValueError('autodetected range of [{}, {}] is not finite'.format(first_edge, last_edge))
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5
    return (first_edge, last_edge)