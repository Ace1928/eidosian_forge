from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
def _weighted_mean_absolute_percentage_error_compute(sum_abs_error: Tensor, sum_scale: Tensor, epsilon: float=1.17e-06) -> Tensor:
    """Compute Weighted Absolute Percentage Error.

    Args:
        sum_abs_error: scalar with sum of absolute errors
        sum_scale: scalar with sum of target values
        epsilon: small float to prevent division by zero

    """
    return sum_abs_error / torch.clamp(sum_scale, min=epsilon)