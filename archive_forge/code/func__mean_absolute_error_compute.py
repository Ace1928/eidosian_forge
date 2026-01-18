from typing import Tuple, Union
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
def _mean_absolute_error_compute(sum_abs_error: Tensor, num_obs: Union[int, Tensor]) -> Tensor:
    """Compute Mean Absolute Error.

    Args:
        sum_abs_error: Sum of absolute value of errors over all observations
        num_obs: Number of predictions or observations

    Example:
        >>> preds = torch.tensor([0., 1, 2, 3])
        >>> target = torch.tensor([0., 1, 2, 2])
        >>> sum_abs_error, num_obs = _mean_absolute_error_update(preds, target)
        >>> _mean_absolute_error_compute(sum_abs_error, num_obs)
        tensor(0.2500)

    """
    return sum_abs_error / num_obs