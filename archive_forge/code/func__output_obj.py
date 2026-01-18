import functools
import numbers
import operator
import numpy
import cupy
from cupy import _core
from cupy._creation import from_data
from cupy._manipulation import join
def _output_obj(self, obj, ndim, ndmin, trans1d):
    k2 = ndmin - ndim
    if trans1d < 0:
        trans1d += k2 + 1
    defaxes = list(range(ndmin))
    k1 = trans1d
    axes = defaxes[:k1] + defaxes[k2:] + defaxes[k1:k2]
    return obj.transpose(axes)