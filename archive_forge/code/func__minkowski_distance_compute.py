import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.exceptions import TorchMetricsUserError
def _minkowski_distance_compute(distance: Tensor, p: float) -> Tensor:
    """Compute Minkowski Distance.

    Args:
        distance: Sum of the p-th powers of errors over all observations
        p: The non-negative numeric power the errors are to be raised to

    Example:
        >>> preds = torch.tensor([0., 1, 2, 3])
        >>> target = torch.tensor([0., 2, 3, 1])
        >>> distance_p_sum = _minkowski_distance_update(preds, target, 5)
        >>> _minkowski_distance_compute(distance_p_sum, 5)
        tensor(2.0244)

    """
    return torch.pow(distance, 1.0 / p)