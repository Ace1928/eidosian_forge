from typing import Tuple, Union
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
def _mean_squared_error_update(preds: Tensor, target: Tensor, num_outputs: int) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute Mean Squared Error.

    Check for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        num_outputs: Number of outputs in multioutput setting

    """
    _check_same_shape(preds, target)
    if num_outputs == 1:
        preds = preds.view(-1)
        target = target.view(-1)
    diff = preds - target
    sum_squared_error = torch.sum(diff * diff, dim=0)
    return (sum_squared_error, target.shape[0])