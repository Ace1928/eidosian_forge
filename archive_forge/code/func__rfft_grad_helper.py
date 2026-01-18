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
def _rfft_grad_helper(rank, irfft_fn):
    """Returns a gradient function for an RFFT of the provided rank."""
    assert rank in (1, 2), 'Gradient for RFFT3D is not implemented.'

    def _grad(op, grad):
        """A gradient function for RFFT with the provided `rank` and `irfft_fn`."""
        fft_length = op.inputs[1]
        complex_dtype = grad.dtype
        real_dtype = complex_dtype.real_dtype
        input_shape = _array_ops.shape(op.inputs[0])
        is_even = _math_ops.cast(1 - fft_length[-1] % 2, complex_dtype)

        def _tile_for_broadcasting(matrix, t):
            expanded = _array_ops.reshape(matrix, _array_ops.concat([_array_ops.ones([_array_ops.rank(t) - 2], _dtypes.int32), _array_ops.shape(matrix)], 0))
            return _array_ops.tile(expanded, _array_ops.concat([_array_ops.shape(t)[:-2], [1, 1]], 0))

        def _mask_matrix(length):
            """Computes t_n = exp(sqrt(-1) * pi * n^2 / line_len)."""
            a = _array_ops.tile(_array_ops.expand_dims(_math_ops.range(length), 0), (length, 1))
            b = _array_ops.transpose(a, [1, 0])
            return _math_ops.exp(-2j * np.pi * _math_ops.cast(a * b, complex_dtype) / _math_ops.cast(length, complex_dtype))

        def _ymask(length):
            """A sequence of [1+0j, -1+0j, 1+0j, -1+0j, ...] with length `length`."""
            return _math_ops.cast(1 - 2 * (_math_ops.range(length) % 2), complex_dtype)
        y0 = grad[..., 0:1]
        if rank == 1:
            ym = grad[..., -1:]
            extra_terms = y0 + is_even * ym * _ymask(input_shape[-1])
        elif rank == 2:
            base_mask = _mask_matrix(input_shape[-2])
            tiled_mask = _tile_for_broadcasting(base_mask, y0)
            y0_term = _math_ops.matmul(tiled_mask, _math_ops.conj(y0))
            extra_terms = y0_term
            ym = grad[..., -1:]
            ym_term = _math_ops.matmul(tiled_mask, _math_ops.conj(ym))
            inner_dim = input_shape[-1]
            ym_term = _array_ops.tile(ym_term, _array_ops.concat([_array_ops.ones([_array_ops.rank(grad) - 1], _dtypes.int32), [inner_dim]], 0)) * _ymask(inner_dim)
            extra_terms += is_even * ym_term
        input_size = _math_ops.cast(_fft_size_for_grad(op.inputs[0], rank), real_dtype)
        the_irfft = irfft_fn(grad, fft_length)
        return (0.5 * (the_irfft * input_size + _math_ops.real(extra_terms)), None)
    return _grad