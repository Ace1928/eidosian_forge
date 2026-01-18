import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torchmetrics.utilities.exceptions import TorchMetricsUserWarning
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_12, _TORCH_GREATER_EQUAL_1_13, _XLA_AVAILABLE
from torchmetrics.utilities.prints import rank_zero_warn
def _flexible_bincount(x: Tensor) -> Tensor:
    """Similar to `_bincount`, but works also with tensor that do not contain continuous values.

    Args:
        x: tensor to count

    Returns:
        Number of occurrences for each unique element in x

    """
    x = x - x.min()
    unique_x = torch.unique(x)
    output = _bincount(x, minlength=torch.max(unique_x) + 1)
    return output[unique_x]