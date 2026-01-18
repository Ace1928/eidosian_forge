from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
def _multiclass_calibration_error_arg_validation(num_classes: int, n_bins: int, norm: Literal['l1', 'l2', 'max']='l1', ignore_index: Optional[int]=None) -> None:
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(f'Expected argument `num_classes` to be an integer larger than 1, but got {num_classes}')
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError(f'Expected argument `n_bins` to be an integer larger than 0, but got {n_bins}')
    allowed_norm = ('l1', 'l2', 'max')
    if norm not in allowed_norm:
        raise ValueError(f'Expected argument `norm` to be one of {allowed_norm}, but got {norm}.')
    if ignore_index is not None and (not isinstance(ignore_index, int)):
        raise ValueError(f'Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}')