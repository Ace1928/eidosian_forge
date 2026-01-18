from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide, interp
from torchmetrics.utilities.data import _bincount, _cumsum
from torchmetrics.utilities.enums import ClassificationTask
def _multiclass_precision_recall_curve_update(preds: Tensor, target: Tensor, num_classes: int, thresholds: Optional[Tensor], average: Optional[Literal['micro', 'macro']]=None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Return the state to calculate the pr-curve with.

    If thresholds is `None` the direct preds and targets are used. If thresholds is not `None` we compute a multi
    threshold confusion matrix.

    """
    if thresholds is None:
        return (preds, target)
    if average == 'micro':
        return _binary_precision_recall_curve_update(preds, target, thresholds)
    if preds.numel() * num_classes <= 1000000:
        update_fn = _multiclass_precision_recall_curve_update_vectorized
    else:
        update_fn = _multiclass_precision_recall_curve_update_loop
    return update_fn(preds, target, num_classes, thresholds)