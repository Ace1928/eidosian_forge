from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.functional.classification.roc import (
from torchmetrics.utilities.compute import _auc_compute_without_check, _safe_divide
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def _multilabel_auroc_arg_validation(num_labels: int, average: Optional[Literal['micro', 'macro', 'weighted', 'none']], thresholds: Optional[Union[int, List[float], Tensor]]=None, ignore_index: Optional[int]=None) -> None:
    _multilabel_precision_recall_curve_arg_validation(num_labels, thresholds, ignore_index)
    allowed_average = ('micro', 'macro', 'weighted', 'none', None)
    if average not in allowed_average:
        raise ValueError(f'Expected argument `average` to be one of {allowed_average} but got {average}')