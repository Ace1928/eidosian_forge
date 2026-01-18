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
def _vectorize_call_with_signature(self, func, args):
    """Vectorized call over positional arguments with a signature."""
    input_core_dims, output_core_dims = self._in_and_out_core_dims
    if len(args) != len(input_core_dims):
        raise TypeError('wrong number of positional arguments: expected %r, got %r' % (len(input_core_dims), len(args)))
    args = tuple((asanyarray(arg) for arg in args))
    broadcast_shape, dim_sizes = _parse_input_dimensions(args, input_core_dims)
    input_shapes = _calculate_shapes(broadcast_shape, dim_sizes, input_core_dims)
    args = [np.broadcast_to(arg, shape, subok=True) for arg, shape in zip(args, input_shapes)]
    outputs = None
    otypes = self.otypes
    nout = len(output_core_dims)
    for index in np.ndindex(*broadcast_shape):
        results = func(*(arg[index] for arg in args))
        n_results = len(results) if isinstance(results, tuple) else 1
        if nout != n_results:
            raise ValueError('wrong number of outputs from pyfunc: expected %r, got %r' % (nout, n_results))
        if nout == 1:
            results = (results,)
        if outputs is None:
            for result, core_dims in zip(results, output_core_dims):
                _update_dim_sizes(dim_sizes, result, core_dims)
            outputs = _create_arrays(broadcast_shape, dim_sizes, output_core_dims, otypes, results)
        for output, result in zip(outputs, results):
            output[index] = result
    if outputs is None:
        if otypes is None:
            raise ValueError('cannot call `vectorize` on size 0 inputs unless `otypes` is set')
        if builtins.any((dim not in dim_sizes for dims in output_core_dims for dim in dims)):
            raise ValueError('cannot call `vectorize` with a signature including new output dimensions on size 0 inputs')
        outputs = _create_arrays(broadcast_shape, dim_sizes, output_core_dims, otypes)
    return outputs[0] if nout == 1 else outputs