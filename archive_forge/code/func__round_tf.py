from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _round_tf(tensor, decimals=0):
    """Implement a TensorFlow version of np.round"""
    tf = _i('tf')
    tol = 10 ** decimals
    return tf.round(tensor * tol) / tol