from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.prints import rank_zero_warn
def _compute_rows_and_cols_corrected(num_rows: int, num_cols: int, confmat_sum: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute bias-corrected number of rows and columns."""
    rows_corrected = num_rows - (num_rows - 1) ** 2 / (confmat_sum - 1)
    cols_corrected = num_cols - (num_cols - 1) ** 2 / (confmat_sum - 1)
    return (rows_corrected, cols_corrected)