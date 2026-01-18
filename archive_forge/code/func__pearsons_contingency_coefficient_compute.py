import itertools
from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import _multiclass_confusion_matrix_update
from torchmetrics.functional.nominal.utils import (
def _pearsons_contingency_coefficient_compute(confmat: Tensor) -> Tensor:
    """Compute Pearson's Contingency Coefficient based on a pre-computed confusion matrix.

    Args:
        confmat: Confusion matrix for observed data

    Returns:
        Pearson's Contingency Coefficient

    """
    confmat = _drop_empty_rows_and_cols(confmat)
    cm_sum = confmat.sum()
    chi_squared = _compute_chi_squared(confmat, bias_correction=False)
    phi_squared = chi_squared / cm_sum
    tschuprows_t_value = torch.sqrt(phi_squared / (1 + phi_squared))
    return tschuprows_t_value.clamp(0.0, 1.0)