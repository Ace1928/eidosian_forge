from typing import Optional, Tuple
import torch
from torch import Tensor
def _auc_compute(x: Tensor, y: Tensor, reorder: bool=False) -> Tensor:
    with torch.no_grad():
        if reorder:
            x, x_idx = torch.sort(x, stable=True)
            y = y[x_idx]
        dx = x[1:] - x[:-1]
        if (dx < 0).any():
            if (dx <= 0).all():
                direction = -1.0
            else:
                raise ValueError('The `x` tensor is neither increasing or decreasing. Try setting the reorder argument to `True`.')
        else:
            direction = 1.0
        return _auc_compute_without_check(x, y, direction)