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
def _convert_to_spatial_operand(x, num_spatial_dims, data_format='channels_last', include_batch_and_channels=True):
    x = (x,) * num_spatial_dims if isinstance(x, int) else x
    if not include_batch_and_channels:
        return x
    if data_format == 'channels_last':
        x = (1,) + x + (1,)
    else:
        x = (1,) + (1,) + x
    return x