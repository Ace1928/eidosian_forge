from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide, interp
from torchmetrics.utilities.data import _bincount, _cumsum
from torchmetrics.utilities.enums import ClassificationTask
def _binary_precision_recall_curve_compute(state: Union[Tensor, Tuple[Tensor, Tensor]], thresholds: Optional[Tensor], pos_label: int=1) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute the final pr-curve.

    If state is a single tensor, then we calculate the pr-curve from a multi threshold confusion matrix. If state is
    original input, then we dynamically compute the binary classification curve.

    """
    if isinstance(state, Tensor) and thresholds is not None:
        tps = state[:, 1, 1]
        fps = state[:, 0, 1]
        fns = state[:, 1, 0]
        precision = _safe_divide(tps, tps + fps)
        recall = _safe_divide(tps, tps + fns)
        precision = torch.cat([precision, torch.ones(1, dtype=precision.dtype, device=precision.device)])
        recall = torch.cat([recall, torch.zeros(1, dtype=recall.dtype, device=recall.device)])
        return (precision, recall, thresholds)
    fps, tps, thresholds = _binary_clf_curve(state[0], state[1], pos_label=pos_label)
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    precision = torch.cat([precision.flip(0), torch.ones(1, dtype=precision.dtype, device=precision.device)])
    recall = torch.cat([recall.flip(0), torch.zeros(1, dtype=recall.dtype, device=recall.device)])
    thresholds = thresholds.flip(0).detach().clone()
    return (precision, recall, thresholds)