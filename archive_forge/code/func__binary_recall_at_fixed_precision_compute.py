from typing import Callable, List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.utilities.enums import ClassificationTask
def _binary_recall_at_fixed_precision_compute(state: Union[Tensor, Tuple[Tensor, Tensor]], thresholds: Optional[Tensor], min_precision: float, pos_label: int=1, reduce_fn: Callable=_recall_at_precision) -> Tuple[Tensor, Tensor]:
    precision, recall, thresholds = _binary_precision_recall_curve_compute(state, thresholds, pos_label)
    return reduce_fn(precision, recall, thresholds, min_precision)