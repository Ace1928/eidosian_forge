from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def _multiclass_confusion_matrix_format(preds: Tensor, target: Tensor, ignore_index: Optional[int]=None, convert_to_labels: bool=True) -> Tuple[Tensor, Tensor]:
    """Convert all input to label format.

    - Applies argmax if preds have one more dimension than target
    - Remove all datapoints that should be ignored

    """
    if preds.ndim == target.ndim + 1 and convert_to_labels:
        preds = preds.argmax(dim=1)
    preds = preds.flatten() if convert_to_labels else torch.movedim(preds, 1, -1).reshape(-1, preds.shape[1])
    target = target.flatten()
    if ignore_index is not None:
        idx = target != ignore_index
        preds = preds[idx]
        target = target[idx]
    return (preds, target)