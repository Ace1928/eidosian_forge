from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
def _log_cosh_error_compute(sum_log_cosh_error: Tensor, num_obs: Tensor) -> Tensor:
    """Compute Mean Squared Error.

    Args:
        sum_log_cosh_error: Sum of LogCosh errors over all observations
        num_obs: Number of predictions or observations

    """
    return (sum_log_cosh_error / num_obs).squeeze()