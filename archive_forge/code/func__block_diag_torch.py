from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _block_diag_torch(tensors):
    """Torch implementation of scipy.linalg.block_diag"""
    torch = _i('torch')
    sizes = np.array([t.shape for t in tensors])
    shape = np.sum(sizes, axis=0).tolist()
    res = torch.zeros(shape, dtype=tensors[0].dtype, device=tensors[0].device)
    p = np.cumsum(sizes, axis=0)
    ridx, cidx = np.stack([p - sizes, p]).T
    for t, r, c in zip(tensors, ridx, cidx):
        row = np.arange(*r).reshape(-1, 1)
        col = np.arange(*c).reshape(1, -1)
        res[row, col] = t
    return res