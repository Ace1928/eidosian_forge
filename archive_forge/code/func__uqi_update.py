from typing import Optional, Sequence, Tuple
import torch
from torch import Tensor, nn
from typing_extensions import Literal
from torchmetrics.functional.image.utils import _gaussian_kernel_2d
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
def _uqi_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Update and returns variables required to compute Universal Image Quality Index.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    """
    if preds.dtype != target.dtype:
        raise TypeError(f'Expected `preds` and `target` to have the same data type. Got preds: {preds.dtype} and target: {target.dtype}.')
    _check_same_shape(preds, target)
    if len(preds.shape) != 4:
        raise ValueError(f'Expected `preds` and `target` to have BxCxHxW shape. Got preds: {preds.shape} and target: {target.shape}.')
    return (preds, target)