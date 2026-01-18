from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
def _log_cosh_error_update(preds: Tensor, target: Tensor, num_outputs: int) -> Tuple[Tensor, Tensor]:
    """Update and returns variables required to compute LogCosh error.

    Check for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        num_outputs: Number of outputs in multioutput setting

    Return:
        Sum of LogCosh error over examples, and total number of examples

    """
    _check_same_shape(preds, target)
    _check_data_shape_to_num_outputs(preds, target, num_outputs)
    preds, target = _unsqueeze_tensors(preds, target)
    diff = preds - target
    sum_log_cosh_error = torch.log((torch.exp(diff) + torch.exp(-diff)) / 2).sum(0).squeeze()
    num_obs = torch.tensor(target.shape[0], device=preds.device)
    return (sum_log_cosh_error, num_obs)