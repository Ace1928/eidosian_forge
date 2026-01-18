import inspect
import os
from typing import List, NamedTuple, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from typing_extensions import Literal
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_13
def _normalize_tensor(in_feat: Tensor, eps: float=1e-08) -> Tensor:
    """Normalize input tensor."""
    norm_factor = torch.sqrt(eps + torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / norm_factor