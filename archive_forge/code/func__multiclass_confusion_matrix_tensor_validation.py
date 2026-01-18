from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def _multiclass_confusion_matrix_tensor_validation(preds: Tensor, target: Tensor, num_classes: int, ignore_index: Optional[int]=None) -> None:
    """Validate tensor input.

    - if target has one more dimension than preds, then all dimensions except for preds.shape[1] should match
    exactly. preds.shape[1] should have size equal to number of classes
    - if preds and target have same number of dims, then all dimensions should match
    - all values in target tensor that are not ignored have to be {0, ..., num_classes - 1}
    - if pred tensor is not floating point, then all values also have to be in {0, ..., num_classes - 1}

    """
    if preds.ndim == target.ndim + 1:
        if not preds.is_floating_point():
            raise ValueError('If `preds` have one dimension more than `target`, `preds` should be a float tensor.')
        if preds.shape[1] != num_classes:
            raise ValueError('If `preds` have one dimension more than `target`, `preds.shape[1]` should be equal to number of classes.')
        if preds.shape[2:] != target.shape[1:]:
            raise ValueError('If `preds` have one dimension more than `target`, the shape of `preds` should be (N, C, ...), and the shape of `target` should be (N, ...).')
    elif preds.ndim == target.ndim:
        if preds.shape != target.shape:
            raise ValueError('The `preds` and `target` should have the same shape,', f' got `preds` with shape={preds.shape} and `target` with shape={target.shape}.')
    else:
        raise ValueError('Either `preds` and `target` both should have the (same) shape (N, ...), or `target` should be (N, ...) and `preds` should be (N, C, ...).')
    check_value = num_classes if ignore_index is None else num_classes + 1
    for t, name in ((target, 'target'),) + ((preds, 'preds'),) if not preds.is_floating_point() else ():
        num_unique_values = len(torch.unique(t))
        if num_unique_values > check_value:
            raise RuntimeError(f'Detected more unique values in `{name}` than expected. Expected only {check_value} but found {num_unique_values} in `target`.')