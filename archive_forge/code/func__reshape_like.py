import numpy as np
from .. import ndarray
from ..base import numeric_types
from .block import HybridBlock
from ..util import is_np_array
def _reshape_like(F, x, y):
    """Reshapes x to the same shape as y."""
    if F is ndarray:
        return x.reshape(y.shape)
    elif is_np_array():
        F = F.npx
    return F.reshape_like(x, y)