from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide, interp
from torchmetrics.utilities.data import _bincount, _cumsum
from torchmetrics.utilities.enums import ClassificationTask
def _binary_precision_recall_curve_format(preds: Tensor, target: Tensor, thresholds: Optional[Union[int, List[float], Tensor]]=None, ignore_index: Optional[int]=None) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Convert all input to the right format.

    - flattens additional dimensions
    - Remove all datapoints that should be ignored
    - Applies sigmoid if pred tensor not in [0,1] range
    - Format thresholds arg to be a tensor

    """
    preds = preds.flatten()
    target = target.flatten()
    if ignore_index is not None:
        idx = target != ignore_index
        preds = preds[idx]
        target = target[idx]
    if not torch.all((preds >= 0) * (preds <= 1)):
        preds = preds.sigmoid()
    thresholds = _adjust_threshold_arg(thresholds, preds.device)
    return (preds, target, thresholds)