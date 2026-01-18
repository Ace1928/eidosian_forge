import warnings
from collections import namedtuple
from typing import Any, Optional
import torch
def _pad_tensor_for_matmul(self, original_tensor: torch.Tensor) -> torch.Tensor:
    """
        Calculates padding for dense tensor and pads tensor if necessary.
        If padding is not required, this function returns the original tensor.
        """
    assert original_tensor.dim() == 2
    m, n = original_tensor.shape
    min_rows = _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG[original_tensor.dtype].min_rows
    min_cols = _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG[original_tensor.dtype].min_cols
    to_pad_m = -m % min_rows if m < min_rows or m % min_rows else 0
    to_pad_n = -n % min_cols if n < min_cols or n % min_rows else 0
    if to_pad_m or to_pad_n:
        return torch.nn.functional.pad(original_tensor, (0, to_pad_n, 0, to_pad_m))
    else:
        return original_tensor