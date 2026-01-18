from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.prints import rank_zero_warn
def _compute_chi_squared(confmat: Tensor, bias_correction: bool) -> Tensor:
    """Chi-square test of independenc of variables in a confusion matrix table.

    Adapted from: https://github.com/scipy/scipy/blob/v1.9.2/scipy/stats/contingency.py.

    """
    expected_freqs = _compute_expected_freqs(confmat)
    df = expected_freqs.numel() - sum(expected_freqs.shape) + expected_freqs.ndim - 1
    if df == 0:
        return torch.tensor(0.0, device=confmat.device)
    if df == 1 and bias_correction:
        diff = expected_freqs - confmat
        direction = diff.sign()
        confmat += direction * torch.minimum(0.5 * torch.ones_like(direction), direction.abs())
    return torch.sum((confmat - expected_freqs) ** 2 / expected_freqs)