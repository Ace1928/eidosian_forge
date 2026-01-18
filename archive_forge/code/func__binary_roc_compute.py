from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.compute import _safe_divide, interp
from torchmetrics.utilities.enums import ClassificationTask
def _binary_roc_compute(state: Union[Tensor, Tuple[Tensor, Tensor]], thresholds: Optional[Tensor], pos_label: int=1) -> Tuple[Tensor, Tensor, Tensor]:
    if isinstance(state, Tensor) and thresholds is not None:
        tps = state[:, 1, 1]
        fps = state[:, 0, 1]
        fns = state[:, 1, 0]
        tns = state[:, 0, 0]
        tpr = _safe_divide(tps, tps + fns).flip(0)
        fpr = _safe_divide(fps, fps + tns).flip(0)
        thres = thresholds.flip(0)
    else:
        fps, tps, thres = _binary_clf_curve(preds=state[0], target=state[1], pos_label=pos_label)
        tps = torch.cat([torch.zeros(1, dtype=tps.dtype, device=tps.device), tps])
        fps = torch.cat([torch.zeros(1, dtype=fps.dtype, device=fps.device), fps])
        thres = torch.cat([torch.ones(1, dtype=thres.dtype, device=thres.device), thres])
        if fps[-1] <= 0:
            rank_zero_warn('No negative samples in targets, false positive value should be meaningless. Returning zero tensor in false positive score', UserWarning)
            fpr = torch.zeros_like(thres)
        else:
            fpr = fps / fps[-1]
        if tps[-1] <= 0:
            rank_zero_warn('No positive samples in targets, true positive value should be meaningless. Returning zero tensor in true positive score', UserWarning)
            tpr = torch.zeros_like(thres)
        else:
            tpr = tps / tps[-1]
    return (fpr, tpr, thres)