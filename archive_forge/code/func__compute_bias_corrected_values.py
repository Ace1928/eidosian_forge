from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.prints import rank_zero_warn
def _compute_bias_corrected_values(phi_squared: Tensor, num_rows: int, num_cols: int, confmat_sum: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute bias-corrected Phi Squared and number of rows and columns."""
    phi_squared_corrected = _compute_phi_squared_corrected(phi_squared, num_rows, num_cols, confmat_sum)
    rows_corrected, cols_corrected = _compute_rows_and_cols_corrected(num_rows, num_cols, confmat_sum)
    return (phi_squared_corrected, rows_corrected, cols_corrected)