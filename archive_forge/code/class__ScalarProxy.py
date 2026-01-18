import numpy
from cupy._core._dtype import get_dtype
import cupy
from cupy._core import _fusion_thread_local
from cupy._core import core
from cupy._core._scalar import get_typename
class _ScalarProxy(_VariableProxy):
    """An abstracted scalar object passed to the target function.

    Attributes:
        dtype(dtype): The dtype of the array.
        imag(_ArrayProxy): The imaginary part of the array (Not implemented)
        real(_ArrayProxy): The real part of the array (Not implemented)
        ndim(int): The number of dimensions of the array.
    """

    def __repr__(self):
        return '_ScalarProxy({}, dtype={})'.format(self._emit_param_name(), self.dtype)