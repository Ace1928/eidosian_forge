from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _tf_convert_to_tensor(x, **kwargs):
    if isinstance(x, _i('tf').Tensor) and 'dtype' in kwargs:
        return _i('tf').cast(x, **kwargs)
    return _i('tf').convert_to_tensor(x, **kwargs)