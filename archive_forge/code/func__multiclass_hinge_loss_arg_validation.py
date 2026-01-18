from typing import Optional, Tuple
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.data import to_onehot
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
def _multiclass_hinge_loss_arg_validation(num_classes: int, squared: bool=False, multiclass_mode: Literal['crammer-singer', 'one-vs-all']='crammer-singer', ignore_index: Optional[int]=None) -> None:
    _binary_hinge_loss_arg_validation(squared, ignore_index)
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(f'Expected argument `num_classes` to be an integer larger than 1, but got {num_classes}')
    allowed_mm = ('crammer-singer', 'one-vs-all')
    if multiclass_mode not in allowed_mm:
        raise ValueError(f'Expected argument `multiclass_mode` to be one of {allowed_mm}, but got {multiclass_mode}.')