import inspect
import os
from typing import List, NamedTuple, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from typing_extensions import Literal
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_13
def _resize_tensor(x: Tensor, size: int=64) -> Tensor:
    """https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/sample_similarity_lpips.py#L127C22-L132."""
    if x.shape[-1] > size and x.shape[-2] > size:
        return torch.nn.functional.interpolate(x, (size, size), mode='area')
    return torch.nn.functional.interpolate(x, (size, size), mode='bilinear', align_corners=False)