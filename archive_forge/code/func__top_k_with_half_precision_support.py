import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torchmetrics.utilities.exceptions import TorchMetricsUserWarning
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_12, _TORCH_GREATER_EQUAL_1_13, _XLA_AVAILABLE
from torchmetrics.utilities.prints import rank_zero_warn
def _top_k_with_half_precision_support(x: Tensor, k: int=1, dim: int=1) -> Tensor:
    """torch.top_k does not support half precision on CPU."""
    if x.dtype == torch.half and (not x.is_cuda):
        if not _TORCH_GREATER_EQUAL_1_13:
            raise RuntimeError('Half precision (torch.float16) is not supported on CPU for PyTorch < 1.13.')
        idx = torch.argsort(x, dim=dim, stable=True).flip(dim)
        return idx.narrow(dim, 0, k)
    return x.topk(k=k, dim=dim).indices