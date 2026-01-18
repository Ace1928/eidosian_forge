from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _scatter_jax(indices, array, new_dimensions):
    from jax import numpy as jnp
    new_array = jnp.zeros(new_dimensions, dtype=array.dtype.type)
    new_array = new_array.at[indices].set(array)
    return new_array