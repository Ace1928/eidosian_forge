import tensorflow as tf
from tensorflow.experimental import numpy as tfnp
from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.tensorflow.core import cast
from keras.src.backend.tensorflow.core import convert_to_tensor
def _get_complex_tensor_from_tuple(x):
    if not isinstance(x, (tuple, list)) or len(x) != 2:
        raise ValueError(f'Input `x` should be a tuple of two tensors - real and imaginary.Received: x={x}')
    real, imag = x
    real = convert_to_tensor(real)
    imag = convert_to_tensor(imag)
    if real.shape != imag.shape:
        raise ValueError(f'Input `x` should be a tuple of two tensors - real and imaginary.Both the real and imaginary parts should have the same shape. Received: x[0].shape = {real.shape}, x[1].shape = {imag.shape}')
    if not real.dtype.is_floating or not imag.dtype.is_floating:
        raise ValueError(f'At least one tensor in input `x` is not of type float.Received: x={x}.')
    complex_input = tf.dtypes.complex(real, imag)
    return complex_input