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
def _binary_auroc_compute(state: Union[Tensor, Tuple[Tensor, Tensor]], thresholds: Optional[Tensor], max_fpr: Optional[float]=None, pos_label: int=1) -> Tensor:
    fpr, tpr, _ = _binary_roc_compute(state, thresholds, pos_label)
    if max_fpr is None or max_fpr == 1 or fpr.sum() == 0 or (tpr.sum() == 0):
        return _auc_compute_without_check(fpr, tpr, 1.0)
    _device = fpr.device if isinstance(fpr, Tensor) else fpr[0].device
    max_area: Tensor = tensor(max_fpr, device=_device)
    stop = torch.bucketize(max_area, fpr, out_int32=True, right=True)
    weight = (max_area - fpr[stop - 1]) / (fpr[stop] - fpr[stop - 1])
    interp_tpr: Tensor = torch.lerp(tpr[stop - 1], tpr[stop], weight)
    tpr = torch.cat([tpr[:stop], interp_tpr.view(1)])
    fpr = torch.cat([fpr[:stop], max_area.view(1)])
    partial_auc = _auc_compute_without_check(fpr, tpr, 1.0)
    min_area: Tensor = 0.5 * max_area ** 2
    return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))