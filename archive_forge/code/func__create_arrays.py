import collections.abc
import functools
import re
import sys
import warnings
from .._utils import set_module
import numpy as np
import numpy.core.numeric as _nx
from numpy.core import transpose
from numpy.core.numeric import (
from numpy.core.umath import (
from numpy.core.fromnumeric import (
from numpy.core.numerictypes import typecodes
from numpy.core import overrides
from numpy.core.function_base import add_newdoc
from numpy.lib.twodim_base import diag
from numpy.core.multiarray import (
from numpy.core.umath import _add_newdoc_ufunc as add_newdoc_ufunc
import builtins
from numpy.lib.histograms import histogram, histogramdd  # noqa: F401
def _create_arrays(broadcast_shape, dim_sizes, list_of_core_dims, dtypes, results=None):
    """Helper for creating output arrays in vectorize."""
    shapes = _calculate_shapes(broadcast_shape, dim_sizes, list_of_core_dims)
    if dtypes is None:
        dtypes = [None] * len(shapes)
    if results is None:
        arrays = tuple((np.empty(shape=shape, dtype=dtype) for shape, dtype in zip(shapes, dtypes)))
    else:
        arrays = tuple((np.empty_like(result, shape=shape, dtype=dtype) for result, shape, dtype in zip(results, shapes, dtypes)))
    return arrays