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
def _update_dim_sizes(dim_sizes, arg, core_dims):
    """
    Incrementally check and update core dimension sizes for a single argument.

    Arguments
    ---------
    dim_sizes : Dict[str, int]
        Sizes of existing core dimensions. Will be updated in-place.
    arg : ndarray
        Argument to examine.
    core_dims : Tuple[str, ...]
        Core dimensions for this argument.
    """
    if not core_dims:
        return
    num_core_dims = len(core_dims)
    if arg.ndim < num_core_dims:
        raise ValueError('%d-dimensional argument does not have enough dimensions for all core dimensions %r' % (arg.ndim, core_dims))
    core_shape = arg.shape[-num_core_dims:]
    for dim, size in zip(core_dims, core_shape):
        if dim in dim_sizes:
            if size != dim_sizes[dim]:
                raise ValueError('inconsistent size for core dimension %r: %r vs %r' % (dim, size, dim_sizes[dim]))
        else:
            dim_sizes[dim] = size