from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.compute import _safe_divide, interp
from torchmetrics.utilities.enums import ClassificationTask
def _multiclass_roc_compute(state: Union[Tensor, Tuple[Tensor, Tensor]], num_classes: int, thresholds: Optional[Tensor], average: Optional[Literal['micro', 'macro']]=None) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    if average == 'micro':
        return _binary_roc_compute(state, thresholds, pos_label=1)
    if isinstance(state, Tensor) and thresholds is not None:
        tps = state[:, :, 1, 1]
        fps = state[:, :, 0, 1]
        fns = state[:, :, 1, 0]
        tns = state[:, :, 0, 0]
        tpr = _safe_divide(tps, tps + fns).flip(0).T
        fpr = _safe_divide(fps, fps + tns).flip(0).T
        thres = thresholds.flip(0)
        tensor_state = True
    else:
        fpr_list, tpr_list, thres_list = ([], [], [])
        for i in range(num_classes):
            res = _binary_roc_compute((state[0][:, i], state[1]), thresholds=None, pos_label=i)
            fpr_list.append(res[0])
            tpr_list.append(res[1])
            thres_list.append(res[2])
        tensor_state = False
    if average == 'macro':
        thres = thres.repeat(num_classes) if tensor_state else torch.cat(thres_list, dim=0)
        thres = thres.sort(descending=True).values
        mean_fpr = fpr.flatten() if tensor_state else torch.cat(fpr_list, dim=0)
        mean_fpr = mean_fpr.sort().values
        mean_tpr = torch.zeros_like(mean_fpr)
        for i in range(num_classes):
            mean_tpr += interp(mean_fpr, fpr[i] if tensor_state else fpr_list[i], tpr[i] if tensor_state else tpr_list[i])
        mean_tpr /= num_classes
        return (mean_fpr, mean_tpr, thres)
    if tensor_state:
        return (fpr, tpr, thres)
    return (fpr_list, tpr_list, thres_list)