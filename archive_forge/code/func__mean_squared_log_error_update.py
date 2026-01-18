from typing import Tuple, Union
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
def _mean_squared_log_error_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, int]:
    """Return variables required to compute Mean Squared Log Error. Checks for same shape of tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    """
    _check_same_shape(preds, target)
    sum_squared_log_error = torch.sum(torch.pow(torch.log1p(preds) - torch.log1p(target), 2))
    return (sum_squared_log_error, target.numel())