import warnings
import numpy as np
import operator
from numba.core import types, utils, config
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.np.numpy_support import (ufunc_find_matching_loop,
from numba.core.errors import (TypingError, NumbaPerformanceWarning,
from numba import pndindex
@classmethod
def _handle_inputs(cls, ufunc, args, kws):
    """
        Process argument types to a given *ufunc*.
        Returns a (base types, explicit outputs, ndims, layout) tuple where:
        - `base types` is a tuple of scalar types for each input
        - `explicit outputs` is a tuple of explicit output types (arrays)
        - `ndims` is the number of dimensions of the loop and also of
          any outputs, explicit or implicit
        - `layout` is the layout for any implicit output to be allocated
        """
    nin = ufunc.nin
    nout = ufunc.nout
    nargs = ufunc.nargs
    assert nargs == nin + nout
    if len(args) < nin:
        msg = "ufunc '{0}': not enough arguments ({1} found, {2} required)"
        raise TypingError(msg=msg.format(ufunc.__name__, len(args), nin))
    if len(args) > nargs:
        msg = "ufunc '{0}': too many arguments ({1} found, {2} maximum)"
        raise TypingError(msg=msg.format(ufunc.__name__, len(args), nargs))
    args = [a.as_array if isinstance(a, types.ArrayCompatible) else a for a in args]
    arg_ndims = [a.ndim if isinstance(a, types.ArrayCompatible) else 0 for a in args]
    ndims = max(arg_ndims)
    explicit_outputs = args[nin:]
    if not all((d == ndims for d in arg_ndims[nin:])):
        msg = "ufunc '{0}' called with unsuitable explicit output arrays."
        raise TypingError(msg=msg.format(ufunc.__name__))
    if not all((isinstance(output, types.ArrayCompatible) for output in explicit_outputs)):
        msg = "ufunc '{0}' called with an explicit output that is not an array"
        raise TypingError(msg=msg.format(ufunc.__name__))
    if not all((output.mutable for output in explicit_outputs)):
        msg = "ufunc '{0}' called with an explicit output that is read-only"
        raise TypingError(msg=msg.format(ufunc.__name__))
    base_types = [x.dtype if isinstance(x, types.ArrayCompatible) else x for x in args]
    layout = None
    if ndims > 0 and len(explicit_outputs) < ufunc.nout:
        layout = 'C'
        layouts = [x.layout if isinstance(x, types.ArrayCompatible) else '' for x in args]
        if 'C' not in layouts and 'F' in layouts:
            layout = 'F'
    return (base_types, explicit_outputs, ndims, layout)