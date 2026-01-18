from typing import Tuple, Union
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
def _mean_squared_error_compute(sum_squared_error: Tensor, num_obs: Union[int, Tensor], squared: bool=True) -> Tensor:
    """Compute Mean Squared Error.

    Args:
        sum_squared_error: Sum of square of errors over all observations
        num_obs: Number of predictions or observations
        squared: Returns RMSE value if set to False.

    Example:
        >>> preds = torch.tensor([0., 1, 2, 3])
        >>> target = torch.tensor([0., 1, 2, 2])
        >>> sum_squared_error, num_obs = _mean_squared_error_update(preds, target, num_outputs=1)
        >>> _mean_squared_error_compute(sum_squared_error, num_obs)
        tensor(0.2500)

    """
    return sum_squared_error / num_obs if squared else torch.sqrt(sum_squared_error / num_obs)