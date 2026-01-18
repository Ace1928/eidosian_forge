from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.enums import ClassificationTask
def _multilabel_jaccard_index_arg_validation(num_labels: int, threshold: float=0.5, ignore_index: Optional[int]=None, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='macro') -> None:
    _multilabel_confusion_matrix_arg_validation(num_labels, threshold, ignore_index)
    allowed_average = ('micro', 'macro', 'weighted', 'none', None)
    if average not in allowed_average:
        raise ValueError(f'Expected argument `average` to be one of {allowed_average}, but got {average}.')