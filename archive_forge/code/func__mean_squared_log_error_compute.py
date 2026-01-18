from typing import Tuple, Union
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
def _mean_squared_log_error_compute(sum_squared_log_error: Tensor, num_obs: Union[int, Tensor]) -> Tensor:
    """Compute Mean Squared Log Error.

    Args:
        sum_squared_log_error:
            Sum of square of log errors over all observations ``(log error = log(target) - log(prediction))``
        num_obs: Number of predictions or observations

    Example:
        >>> preds = torch.tensor([0., 1, 2, 3])
        >>> target = torch.tensor([0., 1, 2, 2])
        >>> sum_squared_log_error, num_obs = _mean_squared_log_error_update(preds, target)
        >>> _mean_squared_log_error_compute(sum_squared_log_error, num_obs)
        tensor(0.0207)

    """
    return sum_squared_log_error / num_obs