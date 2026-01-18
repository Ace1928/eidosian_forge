from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide, interp
from torchmetrics.utilities.data import _bincount, _cumsum
from torchmetrics.utilities.enums import ClassificationTask
def _multiclass_precision_recall_curve_update_vectorized(preds: Tensor, target: Tensor, num_classes: int, thresholds: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Return the multi-threshold confusion matrix to calculate the pr-curve with.

    This implementation is vectorized and faster than `_binary_precision_recall_curve_update_loop` for small
    numbers of samples but less memory- and time-efficient for more samples.

    """
    len_t = len(thresholds)
    preds_t = (preds.unsqueeze(-1) >= thresholds.unsqueeze(0).unsqueeze(0)).long()
    target_t = torch.nn.functional.one_hot(target, num_classes=num_classes)
    unique_mapping = preds_t + 2 * target_t.long().unsqueeze(-1)
    unique_mapping += 4 * torch.arange(num_classes, device=preds.device).unsqueeze(0).unsqueeze(-1)
    unique_mapping += 4 * num_classes * torch.arange(len_t, device=preds.device)
    bins = _bincount(unique_mapping.flatten(), minlength=4 * num_classes * len_t)
    return bins.reshape(len_t, num_classes, 2, 2)