from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide, interp
from torchmetrics.utilities.data import _bincount, _cumsum
from torchmetrics.utilities.enums import ClassificationTask
def _multilabel_precision_recall_curve_tensor_validation(preds: Tensor, target: Tensor, num_labels: int, ignore_index: Optional[int]=None) -> None:
    """Validate tensor input.

    - tensors have to be of same shape
    - preds.shape[1] is equal to the number of labels
    - all values in target tensor that are not ignored have to be in {0, 1}
    - that the pred tensor is floating point

    """
    _binary_precision_recall_curve_tensor_validation(preds, target, ignore_index)
    if preds.shape[1] != num_labels:
        raise ValueError(f'Expected both `target.shape[1]` and `preds.shape[1]` to be equal to the number of labels but got {preds.shape[1]} and expected {num_labels}')