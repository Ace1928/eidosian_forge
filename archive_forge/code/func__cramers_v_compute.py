import itertools
from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import _multiclass_confusion_matrix_update
from torchmetrics.functional.nominal.utils import (
def _cramers_v_compute(confmat: Tensor, bias_correction: bool) -> Tensor:
    """Compute Cramers' V statistic based on a pre-computed confusion matrix.

    Args:
        confmat: Confusion matrix for observed data
        bias_correction: Indication of whether to use bias correction.

    Returns:
        Cramer's V statistic

    """
    confmat = _drop_empty_rows_and_cols(confmat)
    cm_sum = confmat.sum()
    chi_squared = _compute_chi_squared(confmat, bias_correction)
    phi_squared = chi_squared / cm_sum
    num_rows, num_cols = confmat.shape
    if bias_correction:
        phi_squared_corrected, rows_corrected, cols_corrected = _compute_bias_corrected_values(phi_squared, num_rows, num_cols, cm_sum)
        if torch.min(rows_corrected, cols_corrected) == 1:
            _unable_to_use_bias_correction_warning(metric_name="Cramer's V")
            return torch.tensor(float('nan'), device=confmat.device)
        cramers_v_value = torch.sqrt(phi_squared_corrected / torch.min(rows_corrected - 1, cols_corrected - 1))
    else:
        cramers_v_value = torch.sqrt(phi_squared / min(num_rows - 1, num_cols - 1))
    return cramers_v_value.clamp(0.0, 1.0)