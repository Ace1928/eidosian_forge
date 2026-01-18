import math as _math
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.ops import math_ops as _math_ops
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _dct_internal(input, type=2, n=None, axis=-1, norm=None, name=None):
    """Computes the 1D Discrete Cosine Transform (DCT) of `input`.

  This internal version of `dct` does not perform any validation and accepts a
  dynamic value for `n` in the form of a rank 0 tensor.

  Args:
    input: A `[..., samples]` `float32`/`float64` `Tensor` containing the
      signals to take the DCT of.
    type: The DCT type to perform. Must be 1, 2, 3 or 4.
    n: The length of the transform. If length is less than sequence length,
      only the first n elements of the sequence are considered for the DCT.
      If n is greater than the sequence length, zeros are padded and then
      the DCT is computed as usual. Can be an int or rank 0 tensor.
    axis: For future expansion. The axis to compute the DCT along. Must be `-1`.
    norm: The normalization to apply. `None` for no normalization or `'ortho'`
      for orthonormal normalization.
    name: An optional name for the operation.

  Returns:
    A `[..., samples]` `float32`/`float64` `Tensor` containing the DCT of
    `input`.
  """
    with _ops.name_scope(name, 'dct', [input]):
        input = _ops.convert_to_tensor(input)
        zero = _ops.convert_to_tensor(0.0, dtype=input.dtype)
        seq_len = tensor_shape.dimension_value(input.shape[-1]) or _array_ops.shape(input)[-1]
        if n is not None:

            def truncate_input():
                return input[..., 0:n]

            def pad_input():
                rank = len(input.shape)
                padding = [[0, 0] for _ in range(rank)]
                padding[rank - 1][1] = n - seq_len
                padding = _ops.convert_to_tensor(padding, dtype=_dtypes.int32)
                return _array_ops.pad(input, paddings=padding)
            input = smart_cond.smart_cond(n <= seq_len, truncate_input, pad_input)
        axis_dim = tensor_shape.dimension_value(input.shape[-1]) or _array_ops.shape(input)[-1]
        axis_dim_float = _math_ops.cast(axis_dim, input.dtype)
        if type == 1:
            dct1_input = _array_ops.concat([input, input[..., -2:0:-1]], axis=-1)
            dct1 = _math_ops.real(fft_ops.rfft(dct1_input))
            return dct1
        if type == 2:
            scale = 2.0 * _math_ops.exp(_math_ops.complex(zero, -_math_ops.range(axis_dim_float) * _math.pi * 0.5 / axis_dim_float))
            dct2 = _math_ops.real(fft_ops.rfft(input, fft_length=[2 * axis_dim])[..., :axis_dim] * scale)
            if norm == 'ortho':
                n1 = 0.5 * _math_ops.rsqrt(axis_dim_float)
                n2 = n1 * _math.sqrt(2.0)
                weights = _array_ops.pad(_array_ops.expand_dims(n1, 0), [[0, axis_dim - 1]], constant_values=n2)
                dct2 *= weights
            return dct2
        elif type == 3:
            if norm == 'ortho':
                n1 = _math_ops.sqrt(axis_dim_float)
                n2 = n1 * _math.sqrt(0.5)
                weights = _array_ops.pad(_array_ops.expand_dims(n1, 0), [[0, axis_dim - 1]], constant_values=n2)
                input *= weights
            else:
                input *= axis_dim_float
            scale = 2.0 * _math_ops.exp(_math_ops.complex(zero, _math_ops.range(axis_dim_float) * _math.pi * 0.5 / axis_dim_float))
            dct3 = _math_ops.real(fft_ops.irfft(scale * _math_ops.complex(input, zero), fft_length=[2 * axis_dim]))[..., :axis_dim]
            return dct3
        elif type == 4:
            dct2 = _dct_internal(input, type=2, n=2 * axis_dim, axis=axis, norm=None)
            dct4 = dct2[..., 1::2]
            if norm == 'ortho':
                dct4 *= _math.sqrt(0.5) * _math_ops.rsqrt(axis_dim_float)
            return dct4