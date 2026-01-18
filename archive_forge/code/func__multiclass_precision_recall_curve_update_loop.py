from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide, interp
from torchmetrics.utilities.data import _bincount, _cumsum
from torchmetrics.utilities.enums import ClassificationTask
def _multiclass_precision_recall_curve_update_loop(preds: Tensor, target: Tensor, num_classes: int, thresholds: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Return the state to calculate the pr-curve with.

    This implementation loops over thresholds and is more memory-efficient than
    `_binary_precision_recall_curve_update_vectorized`. However, it is slowwer for small
    numbers of samples.

    """
    len_t = len(thresholds)
    target_t = torch.nn.functional.one_hot(target, num_classes=num_classes)
    confmat = thresholds.new_empty((len_t, num_classes, 2, 2), dtype=torch.int64)
    for i in range(len_t):
        preds_t = preds >= thresholds[i]
        confmat[i, :, 1, 1] = (target_t & preds_t).sum(dim=0)
        confmat[i, :, 0, 1] = (~target_t & preds_t).sum(dim=0)
        confmat[i, :, 1, 0] = (target_t & ~preds_t).sum(dim=0)
    confmat[:, :, 0, 0] = len(preds_t) - confmat[:, :, 0, 1] - confmat[:, :, 1, 0] - confmat[:, :, 1, 1]
    return confmat