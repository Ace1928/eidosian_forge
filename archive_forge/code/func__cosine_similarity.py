import numpy as np
from .. import ndarray
from ..base import numeric_types
from .block import HybridBlock
from ..util import is_np_array
def _cosine_similarity(self, F, x, y, axis=-1):
    x_norm = F.norm(x, axis=axis).reshape((-1, 1))
    y_norm = F.norm(y, axis=axis).reshape((-1, 1))
    x_dot_y = F.sum(x * y, axis=axis).reshape((-1, 1))
    if F is ndarray:
        eps_arr = F.array([1e-12])
    else:
        eps_arr = F.full((1, 1), 1e-12)
    return x_dot_y / F.broadcast_maximum(x_norm * y_norm, eps_arr)