import math
import numbers
import operator
import numpy
import cupy
from cupy import _core
from cupy.fft._fft import _cook_shape
from cupyx.scipy.fft import _fft
def _init_nd_shape_and_axes(x, shape, axes):
    """Handles shape and axes arguments for nd transforms."""
    noshape = shape is None
    noaxes = axes is None
    if not noaxes:
        axes = _iterable_of_int(axes, 'axes')
        axes = [a + x.ndim if a < 0 else a for a in axes]
        if any((a >= x.ndim or a < 0 for a in axes)):
            raise ValueError('axes exceeds dimensionality of input')
        if len(set(axes)) != len(axes):
            raise ValueError('all axes must be unique')
    if not noshape:
        shape = _iterable_of_int(shape, 'shape')
        nshape = len(shape)
        if axes and len(axes) != nshape:
            raise ValueError('when given, axes and shape arguments have to be of the same length')
        if noaxes:
            if nshape > x.ndim:
                raise ValueError('shape requires more axes than are present')
            axes = range(x.ndim - len(shape), x.ndim)
        shape = [x.shape[a] if s == -1 else s for s, a in zip(shape, axes)]
    elif noaxes:
        shape = list(x.shape)
        axes = range(x.ndim)
    else:
        shape = [x.shape[a] for a in axes]
    if any((s < 1 for s in shape)):
        raise ValueError(f'invalid number of data points ({shape}) specified')
    return (shape, axes)