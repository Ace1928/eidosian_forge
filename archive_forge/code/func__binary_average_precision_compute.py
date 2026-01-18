from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def _binary_average_precision_compute(state: Union[Tensor, Tuple[Tensor, Tensor]], thresholds: Optional[Tensor]) -> Tensor:
    precision, recall, _ = _binary_precision_recall_curve_compute(state, thresholds)
    return -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])