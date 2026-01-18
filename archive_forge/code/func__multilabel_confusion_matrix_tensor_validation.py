from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def _multilabel_confusion_matrix_tensor_validation(preds: Tensor, target: Tensor, num_labels: int, ignore_index: Optional[int]=None) -> None:
    """Validate tensor input.

    - tensors have to be of same shape
    - the second dimension of both tensors need to be equal to the number of labels
    - all values in target tensor that are not ignored have to be in {0, 1}
    - if pred tensor is not floating point, then all values also have to be in {0, 1}

    """
    _check_same_shape(preds, target)
    if preds.shape[1] != num_labels:
        raise ValueError(f'Expected both `target.shape[1]` and `preds.shape[1]` to be equal to the number of labels but got {preds.shape[1]} and expected {num_labels}')
    unique_values = torch.unique(target)
    if ignore_index is None:
        check = torch.any((unique_values != 0) & (unique_values != 1))
    else:
        check = torch.any((unique_values != 0) & (unique_values != 1) & (unique_values != ignore_index))
    if check:
        raise RuntimeError(f'Detected the following values in `target`: {unique_values} but expected only the following values {([0, 1] if ignore_index is None else [ignore_index])}.')
    if not preds.is_floating_point():
        unique_values = torch.unique(preds)
        if torch.any((unique_values != 0) & (unique_values != 1)):
            raise RuntimeError(f'Detected the following values in `preds`: {unique_values} but expected only the following values [0,1] since preds is a label tensor.')