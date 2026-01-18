from typing import Tuple, Union
import torch
from torch import Tensor
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _check_same_shape
def _r2_score_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor, Tensor, int]:
    """Update and returns variables required to compute R2 score.

    Check for same shape and 1D/2D input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    """
    _check_same_shape(preds, target)
    if preds.ndim > 2:
        raise ValueError(f'Expected both prediction and target to be 1D or 2D tensors, but received tensors with dimension {preds.shape}')
    sum_obs = torch.sum(target, dim=0)
    sum_squared_obs = torch.sum(target * target, dim=0)
    residual = target - preds
    rss = torch.sum(residual * residual, dim=0)
    return (sum_squared_obs, sum_obs, rss, target.size(0))