import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.functional.classification.roc import (
from torchmetrics.utilities.enums import ClassificationTask
def _multiclass_specificity_at_sensitivity_compute(state: Union[Tensor, Tuple[Tensor, Tensor]], num_classes: int, thresholds: Optional[Tensor], min_sensitivity: float) -> Tuple[Tensor, Tensor]:
    fpr, sensitivity, thresholds = _multiclass_roc_compute(state, num_classes, thresholds)
    specificity = [_convert_fpr_to_specificity(fpr_) for fpr_ in fpr]
    if isinstance(state, Tensor):
        res = [_specificity_at_sensitivity(sp, sn, thresholds, min_sensitivity) for sp, sn in zip(specificity, sensitivity)]
    else:
        res = [_specificity_at_sensitivity(sp, sn, t, min_sensitivity) for sp, sn, t in zip(specificity, sensitivity, thresholds)]
    specificity = torch.stack([r[0] for r in res])
    thresholds = torch.stack([r[1] for r in res])
    return (specificity, thresholds)