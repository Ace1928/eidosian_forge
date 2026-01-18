from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _scatter_numpy(indices, array, shape):
    new_array = np.zeros(shape, dtype=array.dtype.type)
    new_array[indices] = array
    return new_array