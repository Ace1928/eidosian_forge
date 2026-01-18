from typing import Any, Dict, List, Optional, Sequence, Union
import torch
from torch import Tensor
from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator
from torchmetrics.functional.detection.iou import _iou_compute, _iou_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def _get_safe_item_values(self, boxes: Tensor) -> Tensor:
    from torchvision.ops import box_convert
    boxes = _fix_empty_tensors(boxes)
    if boxes.numel() > 0:
        boxes = box_convert(boxes, in_fmt=self.box_format, out_fmt='xyxy')
    return boxes