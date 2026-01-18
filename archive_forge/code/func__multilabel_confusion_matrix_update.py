from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def _multilabel_confusion_matrix_update(preds: Tensor, target: Tensor, num_labels: int) -> Tensor:
    """Compute the bins to update the confusion matrix with."""
    unique_mapping = (2 * target + preds + 4 * torch.arange(num_labels, device=preds.device)).flatten()
    unique_mapping = unique_mapping[unique_mapping >= 0]
    bins = _bincount(unique_mapping, minlength=4 * num_labels)
    return bins.reshape(num_labels, 2, 2)