from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide, interp
from torchmetrics.utilities.data import _bincount, _cumsum
from torchmetrics.utilities.enums import ClassificationTask
def _multilabel_precision_recall_curve_update(preds: Tensor, target: Tensor, num_labels: int, thresholds: Optional[Tensor]) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Return the state to calculate the pr-curve with.

    If thresholds is `None` the direct preds and targets are used. If thresholds is not `None` we compute a multi
    threshold confusion matrix.

    """
    if thresholds is None:
        return (preds, target)
    len_t = len(thresholds)
    preds_t = (preds.unsqueeze(-1) >= thresholds.unsqueeze(0).unsqueeze(0)).long()
    unique_mapping = preds_t + 2 * target.long().unsqueeze(-1)
    unique_mapping += 4 * torch.arange(num_labels, device=preds.device).unsqueeze(0).unsqueeze(-1)
    unique_mapping += 4 * num_labels * torch.arange(len_t, device=preds.device)
    unique_mapping = unique_mapping[unique_mapping >= 0]
    bins = _bincount(unique_mapping, minlength=4 * num_labels * len_t)
    return bins.reshape(len_t, num_labels, 2, 2)