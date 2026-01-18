from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _scatter_element_add_torch(tensor, index, value):
    """In-place addition of a multidimensional value over various
    indices of a tensor. Note that Torch only supports index assignments
    on non-leaf nodes; if the node is a leaf, we must clone it first."""
    if tensor.is_leaf:
        tensor = tensor.clone()
    tensor[tuple(index)] += value
    return tensor