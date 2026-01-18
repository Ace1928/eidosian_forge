from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.data import _bincount, select_topk
from torchmetrics.utilities.enums import AverageMethod, ClassificationTask, DataType, MDMCAverageMethod
def _binary_stat_scores_format(preds: Tensor, target: Tensor, threshold: float=0.5, ignore_index: Optional[int]=None) -> Tuple[Tensor, Tensor]:
    """Convert all input to label format.

    - If preds tensor is floating point, applies sigmoid if pred tensor not in [0,1] range
    - If preds tensor is floating point, thresholds afterwards
    - Mask all datapoints that should be ignored with negative values

    """
    if preds.is_floating_point():
        if not torch.all((preds >= 0) * (preds <= 1)):
            preds = preds.sigmoid()
        preds = preds > threshold
    preds = preds.reshape(preds.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    if ignore_index is not None:
        idx = target == ignore_index
        target = target.clone()
        target[idx] = -1
    return (preds, target)