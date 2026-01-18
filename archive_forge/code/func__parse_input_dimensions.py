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
def _parse_input_dimensions(args, input_core_dims):
    """
    Parse broadcast and core dimensions for vectorize with a signature.

    Arguments
    ---------
    args : Tuple[ndarray, ...]
        Tuple of input arguments to examine.
    input_core_dims : List[Tuple[str, ...]]
        List of core dimensions corresponding to each input.

    Returns
    -------
    broadcast_shape : Tuple[int, ...]
        Common shape to broadcast all non-core dimensions to.
    dim_sizes : Dict[str, int]
        Common sizes for named core dimensions.
    """
    broadcast_args = []
    dim_sizes = {}
    for arg, core_dims in zip(args, input_core_dims):
        _update_dim_sizes(dim_sizes, arg, core_dims)
        ndim = arg.ndim - len(core_dims)
        dummy_array = np.lib.stride_tricks.as_strided(0, arg.shape[:ndim])
        broadcast_args.append(dummy_array)
    broadcast_shape = np.lib.stride_tricks._broadcast_shape(*broadcast_args)
    return (broadcast_shape, dim_sizes)