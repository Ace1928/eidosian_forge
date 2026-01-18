import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.exceptions import TorchMetricsUserError
def _minkowski_distance_update(preds: Tensor, targets: Tensor, p: float) -> Tensor:
    """Update and return variables required to compute Minkowski distance.

    Checks for same shape of input tensors.

    Args:
        preds: Predicted tensor
        targets: Ground truth tensor
        p: Non-negative number acting as the p to the errors

    """
    _check_same_shape(preds, targets)
    if not (isinstance(p, (float, int)) and p >= 1):
        raise TorchMetricsUserError(f'Argument ``p`` must be a float or int greater than 1, but got {p}')
    difference = torch.abs(preds - targets)
    return torch.sum(torch.pow(difference, p))