from typing import Any, Dict, List, Optional, Sequence, Union
import torch
from torch import Tensor
from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator
from torchmetrics.functional.detection.iou import _iou_compute, _iou_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def _get_gt_classes(self) -> List:
    """Returns a list of unique classes found in ground truth and detection data."""
    if len(self.groundtruth_labels) > 0:
        return torch.cat(self.groundtruth_labels).unique().tolist()
    return []