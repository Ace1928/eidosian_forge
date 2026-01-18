from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide, interp
from torchmetrics.utilities.data import _bincount, _cumsum
from torchmetrics.utilities.enums import ClassificationTask
def _multilabel_precision_recall_curve_compute(state: Union[Tensor, Tuple[Tensor, Tensor]], num_labels: int, thresholds: Optional[Tensor], ignore_index: Optional[int]=None) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    """Compute the final pr-curve.

    If state is a single tensor, then we calculate the pr-curve from a multi threshold confusion matrix. If state is
    original input, then we dynamically compute the binary classification curve in an iterative way.

    """
    if isinstance(state, Tensor) and thresholds is not None:
        tps = state[:, :, 1, 1]
        fps = state[:, :, 0, 1]
        fns = state[:, :, 1, 0]
        precision = _safe_divide(tps, tps + fps)
        recall = _safe_divide(tps, tps + fns)
        precision = torch.cat([precision, torch.ones(1, num_labels, dtype=precision.dtype, device=precision.device)])
        recall = torch.cat([recall, torch.zeros(1, num_labels, dtype=recall.dtype, device=recall.device)])
        return (precision.T, recall.T, thresholds)
    precision_list, recall_list, thres_list = ([], [], [])
    for i in range(num_labels):
        preds = state[0][:, i]
        target = state[1][:, i]
        if ignore_index is not None:
            idx = target == ignore_index
            preds = preds[~idx]
            target = target[~idx]
        res = _binary_precision_recall_curve_compute((preds, target), thresholds=None, pos_label=1)
        precision_list.append(res[0])
        recall_list.append(res[1])
        thres_list.append(res[2])
    return (precision_list, recall_list, thres_list)