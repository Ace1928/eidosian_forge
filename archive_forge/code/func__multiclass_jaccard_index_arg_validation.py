from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.enums import ClassificationTask
def _multiclass_jaccard_index_arg_validation(num_classes: int, ignore_index: Optional[int]=None, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]=None) -> None:
    _multiclass_confusion_matrix_arg_validation(num_classes, ignore_index)
    allowed_average = ('micro', 'macro', 'weighted', 'none', None)
    if average not in allowed_average:
        raise ValueError(f'Expected argument `average` to be one of {allowed_average}, but got {average}.')