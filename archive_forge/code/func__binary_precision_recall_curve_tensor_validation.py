from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide, interp
from torchmetrics.utilities.data import _bincount, _cumsum
from torchmetrics.utilities.enums import ClassificationTask
def _binary_precision_recall_curve_tensor_validation(preds: Tensor, target: Tensor, ignore_index: Optional[int]=None) -> None:
    """Validate tensor input.

    - tensors have to be of same shape
    - all values in target tensor that are not ignored have to be in {0, 1}
    - that the pred tensor is floating point

    """
    _check_same_shape(preds, target)
    if target.is_floating_point():
        raise ValueError(f'Expected argument `target` to be an int or long tensor with ground truth labels but got tensor with dtype {target.dtype}')
    if not preds.is_floating_point():
        raise ValueError(f'Expected argument `preds` to be an floating tensor with probability/logit scores, but got tensor with dtype {preds.dtype}')
    unique_values = torch.unique(target)
    if ignore_index is None:
        check = torch.any((unique_values != 0) & (unique_values != 1))
    else:
        check = torch.any((unique_values != 0) & (unique_values != 1) & (unique_values != ignore_index))
    if check:
        raise RuntimeError(f'Detected the following values in `target`: {unique_values} but expected only the following values {([0, 1] if ignore_index is None else [ignore_index])}.')