from typing import Optional, Tuple
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.data import to_onehot
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
def _binary_hinge_loss_arg_validation(squared: bool, ignore_index: Optional[int]=None) -> None:
    if not isinstance(squared, bool):
        raise ValueError(f'Expected argument `squared` to be an bool but got {squared}')
    if ignore_index is not None and (not isinstance(ignore_index, int)):
        raise ValueError(f'Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}')