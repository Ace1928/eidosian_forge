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
def _convert_to_lax_conv_dimension_numbers(num_spatial_dims, data_format='channels_last', transpose=False):
    """Create a `lax.ConvDimensionNumbers` for the given inputs."""
    num_dims = num_spatial_dims + 2
    if data_format == 'channels_last':
        spatial_dims = tuple(range(1, num_dims - 1))
        inputs_dn = (0, num_dims - 1) + spatial_dims
    else:
        spatial_dims = tuple(range(2, num_dims))
        inputs_dn = (0, 1) + spatial_dims
    if transpose:
        kernel_dn = (num_dims - 2, num_dims - 1) + tuple(range(num_dims - 2))
    else:
        kernel_dn = (num_dims - 1, num_dims - 2) + tuple(range(num_dims - 2))
    return lax.ConvDimensionNumbers(lhs_spec=inputs_dn, rhs_spec=kernel_dn, out_spec=inputs_dn)