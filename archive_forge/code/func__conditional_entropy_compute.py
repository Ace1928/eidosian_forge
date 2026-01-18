import itertools
from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import _multiclass_confusion_matrix_update
from torchmetrics.functional.nominal.utils import (
def _conditional_entropy_compute(confmat: Tensor) -> Tensor:
    """Compute Conditional Entropy Statistic based on a pre-computed confusion matrix.

    .. math::
        H(X|Y) = \\sum_{x, y ~ (X, Y)} p(x, y)\\frac{p(y)}{p(x, y)}

    Args:
        confmat: Confusion matrix for observed data

    Returns:
        Conditional Entropy Value

    """
    confmat = _drop_empty_rows_and_cols(confmat)
    total_occurrences = confmat.sum()
    p_xy_m = confmat / total_occurrences
    p_y = confmat.sum(1) / total_occurrences
    p_y_m = p_y.unsqueeze(1).repeat(1, p_xy_m.shape[1])
    return torch.nansum(p_xy_m * torch.log(p_y_m / p_xy_m))