from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _builtins_ndim(x):
    interface = get_deep_interface(x)
    x = ar.numpy.asarray(x, like=interface)
    return ar.ndim(x)