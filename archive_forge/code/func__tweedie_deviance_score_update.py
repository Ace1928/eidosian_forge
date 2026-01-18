from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_xlogy
def _tweedie_deviance_score_update(preds: Tensor, targets: Tensor, power: float=0.0) -> Tuple[Tensor, Tensor]:
    """Update and returns variables required to compute Deviance Score for the given power.

    Check for same shape of input tensors.

    Args:
        preds: Predicted tensor
        targets: Ground truth tensor
        power: see :func:`tweedie_deviance_score`

    Example:
        >>> targets = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> preds = torch.tensor([4.0, 3.0, 2.0, 1.0])
        >>> _tweedie_deviance_score_update(preds, targets, power=2)
        (tensor(4.8333), tensor(4))

    """
    _check_same_shape(preds, targets)
    zero_tensor = torch.zeros(preds.shape, device=preds.device)
    if 0 < power < 1:
        raise ValueError(f'Deviance Score is not defined for power={power}.')
    if power == 0:
        deviance_score = torch.pow(targets - preds, exponent=2)
    elif power == 1:
        if torch.any(preds <= 0) or torch.any(targets < 0):
            raise ValueError(f"For power={power}, 'preds' has to be strictly positive and 'targets' cannot be negative.")
        deviance_score = 2 * (_safe_xlogy(targets, targets / preds) + preds - targets)
    elif power == 2:
        if torch.any(preds <= 0) or torch.any(targets <= 0):
            raise ValueError(f"For power={power}, both 'preds' and 'targets' have to be strictly positive.")
        deviance_score = 2 * (torch.log(preds / targets) + targets / preds - 1)
    else:
        if power < 0:
            if torch.any(preds <= 0):
                raise ValueError(f"For power={power}, 'preds' has to be strictly positive.")
        elif 1 < power < 2:
            if torch.any(preds <= 0) or torch.any(targets < 0):
                raise ValueError(f"For power={power}, 'targets' has to be strictly positive and 'preds' cannot be negative.")
        elif torch.any(preds <= 0) or torch.any(targets <= 0):
            raise ValueError(f"For power={power}, both 'preds' and 'targets' have to be strictly positive.")
        term_1 = torch.pow(torch.max(targets, zero_tensor), 2 - power) / ((1 - power) * (2 - power))
        term_2 = targets * torch.pow(preds, 1 - power) / (1 - power)
        term_3 = torch.pow(preds, 2 - power) / (2 - power)
        deviance_score = 2 * (term_1 - term_2 + term_3)
    sum_deviance_score = torch.sum(deviance_score)
    num_observations = torch.tensor(torch.numel(deviance_score), device=preds.device)
    return (sum_deviance_score, num_observations)