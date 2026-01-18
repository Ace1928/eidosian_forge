from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def _multiclass_confusion_matrix_update(preds: Tensor, target: Tensor, num_classes: int) -> Tensor:
    """Compute the bins to update the confusion matrix with."""
    unique_mapping = target.to(torch.long) * num_classes + preds.to(torch.long)
    bins = _bincount(unique_mapping, minlength=num_classes ** 2)
    return bins.reshape(num_classes, num_classes)