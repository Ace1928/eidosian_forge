import re
import numpy as np
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import tensor_util as _tensor_util
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.ops import array_ops_stack as _array_ops_stack
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops as _math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _irfft_grad_helper(rank, rfft_fn):
    """Returns a gradient function for an IRFFT of the provided rank."""
    assert rank in (1, 2), 'Gradient for IRFFT3D is not implemented.'

    def _grad(op, grad):
        """A gradient function for IRFFT with the provided `rank` and `rfft_fn`."""
        fft_length = op.inputs[1]
        fft_length_static = _tensor_util.constant_value(fft_length)
        if fft_length_static is not None:
            fft_length = fft_length_static
        real_dtype = grad.dtype
        if real_dtype == _dtypes.float32:
            complex_dtype = _dtypes.complex64
        elif real_dtype == _dtypes.float64:
            complex_dtype = _dtypes.complex128
        is_odd = _math_ops.mod(fft_length[-1], 2)
        input_last_dimension = _array_ops.shape(op.inputs[0])[-1]
        mask = _array_ops.concat([[1.0], 2.0 * _array_ops.ones([input_last_dimension - 2 + is_odd], real_dtype), _array_ops.ones([1 - is_odd], real_dtype)], 0)
        rsize = _math_ops.reciprocal(_math_ops.cast(_fft_size_for_grad(grad, rank), real_dtype))
        the_rfft = rfft_fn(grad, fft_length)
        return (the_rfft * _math_ops.cast(rsize * mask, complex_dtype), None)
    return _grad