import jax
import numpy as np
from jax import lax
from jax import numpy as jnp
from keras.src.backend import standardize_data_format
from keras.src.backend import standardize_dtype
from keras.src.backend.common.backend_utils import (
from keras.src.backend.config import epsilon
from keras.src.backend.numpy.core import cast
from keras.src.backend.numpy.core import convert_to_tensor
from keras.src.backend.numpy.core import is_tensor
from keras.src.utils.module_utils import scipy
def _pool(inputs, initial_value, reduce_fn, pool_size, strides=None, padding='valid'):
    """Helper function to define pooling functions.

    Args:
        inputs: input data of shape `N+2`.
        initial_value: the initial value for the reduction.
        reduce_fn: a reduce function of the form `(T, T) -> T`.
        pool_size: a sequence of `N` integers, representing the window size to
            reduce over.
        strides: a sequence of `N` integers, representing the inter-window
            strides (default: `(1, ..., 1)`).
        padding: either the string `same` or `valid`.

    Returns:
        The output of the reduction for each window slice.
    """
    if padding not in ('same', 'valid'):
        raise ValueError(f"Invalid padding '{padding}', must be 'same' or 'valid'.")
    padding = padding.upper()
    return np.array(lax.reduce_window(inputs, initial_value, reduce_fn, pool_size, strides, padding))