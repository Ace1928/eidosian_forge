from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_xlogy
def _tweedie_deviance_score_compute(sum_deviance_score: Tensor, num_observations: Tensor) -> Tensor:
    """Compute Deviance Score.

    Args:
        sum_deviance_score: Sum of deviance scores accumulated until now.
        num_observations: Number of observations encountered until now.

    Example:
        >>> targets = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> preds = torch.tensor([4.0, 3.0, 2.0, 1.0])
        >>> sum_deviance_score, num_observations = _tweedie_deviance_score_update(preds, targets, power=2)
        >>> _tweedie_deviance_score_compute(sum_deviance_score, num_observations)
        tensor(1.2083)

    """
    return sum_deviance_score / num_observations