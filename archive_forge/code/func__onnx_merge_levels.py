from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.fx
import torchvision
from torch import nn, Tensor
from torchvision.ops.boxes import box_area
from ..utils import _log_api_usage_once
from .roi_align import roi_align
@torch.jit.unused
def _onnx_merge_levels(levels: Tensor, unmerged_results: List[Tensor]) -> Tensor:
    first_result = unmerged_results[0]
    dtype, device = (first_result.dtype, first_result.device)
    res = torch.zeros((levels.size(0), first_result.size(1), first_result.size(2), first_result.size(3)), dtype=dtype, device=device)
    for level in range(len(unmerged_results)):
        index = torch.where(levels == level)[0].view(-1, 1, 1, 1)
        index = index.expand(index.size(0), unmerged_results[level].size(1), unmerged_results[level].size(2), unmerged_results[level].size(3))
        res = res.scatter(0, index, unmerged_results[level])
    return res