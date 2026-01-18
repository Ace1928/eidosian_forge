from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
def _weighted_mean_absolute_percentage_error_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Update and returns variables required to compute Weighted Absolute Percentage Error.

    Check for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    """
    _check_same_shape(preds, target)
    sum_abs_error = (preds - target).abs().sum()
    sum_scale = target.abs().sum()
    return (sum_abs_error, sum_scale)