from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _block_diag_autograd(tensors):
    """Autograd implementation of scipy.linalg.block_diag"""
    _np = _i('qml').numpy
    tensors = [t.reshape((1, len(t))) if len(t.shape) == 1 else t for t in tensors]
    rsizes, csizes = _np.array([t.shape for t in tensors]).T
    all_zeros = [[_np.zeros((rsize, csize)) for csize in csizes] for rsize in rsizes]
    res = _np.hstack([tensors[0], *all_zeros[0][1:]])
    for i, t in enumerate(tensors[1:], start=1):
        row = _np.hstack([*all_zeros[i][:i], t, *all_zeros[i][i + 1:]])
        res = _np.vstack([res, row])
    return res