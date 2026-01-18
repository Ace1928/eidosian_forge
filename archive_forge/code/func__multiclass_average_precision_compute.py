from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def _multiclass_average_precision_compute(state: Union[Tensor, Tuple[Tensor, Tensor]], num_classes: int, average: Optional[Literal['macro', 'weighted', 'none']]='macro', thresholds: Optional[Tensor]=None) -> Tensor:
    precision, recall, _ = _multiclass_precision_recall_curve_compute(state, num_classes, thresholds)
    return _reduce_average_precision(precision, recall, average, weights=_bincount(state[1], minlength=num_classes).float() if thresholds is None else state[0][:, 1, :].sum(-1))