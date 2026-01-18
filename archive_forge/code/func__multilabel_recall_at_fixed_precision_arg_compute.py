from typing import Callable, List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.utilities.enums import ClassificationTask
def _multilabel_recall_at_fixed_precision_arg_compute(state: Union[Tensor, Tuple[Tensor, Tensor]], num_labels: int, thresholds: Optional[Tensor], ignore_index: Optional[int], min_precision: float, reduce_fn: Callable=_recall_at_precision) -> Tuple[Tensor, Tensor]:
    precision, recall, thresholds = _multilabel_precision_recall_curve_compute(state, num_labels, thresholds, ignore_index)
    if isinstance(state, Tensor):
        res = [reduce_fn(p, r, thresholds, min_precision) for p, r in zip(precision, recall)]
    else:
        res = [reduce_fn(p, r, t, min_precision) for p, r, t in zip(precision, recall, thresholds)]
    recall = torch.stack([r[0] for r in res])
    thresholds = torch.stack([r[1] for r in res])
    return (recall, thresholds)