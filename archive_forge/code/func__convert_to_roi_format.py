from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.fx
import torchvision
from torch import nn, Tensor
from torchvision.ops.boxes import box_area
from ..utils import _log_api_usage_once
from .roi_align import roi_align
def _convert_to_roi_format(boxes: List[Tensor]) -> Tensor:
    concat_boxes = torch.cat(boxes, dim=0)
    device, dtype = (concat_boxes.device, concat_boxes.dtype)
    ids = torch.cat([torch.full_like(b[:, :1], i, dtype=dtype, layout=torch.strided, device=device) for i, b in enumerate(boxes)], dim=0)
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois