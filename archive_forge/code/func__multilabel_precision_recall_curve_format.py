from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide, interp
from torchmetrics.utilities.data import _bincount, _cumsum
from torchmetrics.utilities.enums import ClassificationTask
def _multilabel_precision_recall_curve_format(preds: Tensor, target: Tensor, num_labels: int, thresholds: Optional[Union[int, List[float], Tensor]]=None, ignore_index: Optional[int]=None) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Convert all input to the right format.

    - flattens additional dimensions
    - Mask all datapoints that should be ignored with negative values
    - Applies sigmoid if pred tensor not in [0,1] range
    - Format thresholds arg to be a tensor

    """
    preds = preds.transpose(0, 1).reshape(num_labels, -1).T
    target = target.transpose(0, 1).reshape(num_labels, -1).T
    if not torch.all((preds >= 0) * (preds <= 1)):
        preds = preds.sigmoid()
    thresholds = _adjust_threshold_arg(thresholds, preds.device)
    if ignore_index is not None and thresholds is not None:
        preds = preds.clone()
        target = target.clone()
        idx = target == ignore_index
        preds[idx] = -4 * num_labels * (len(thresholds) if thresholds is not None else 1)
        target[idx] = -4 * num_labels * (len(thresholds) if thresholds is not None else 1)
    return (preds, target, thresholds)