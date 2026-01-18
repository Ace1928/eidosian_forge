from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.data import _bincount, select_topk
from torchmetrics.utilities.enums import AverageMethod, ClassificationTask, DataType, MDMCAverageMethod
def _multilabel_stat_scores_compute(tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='macro', multidim_average: Literal['global', 'samplewise']='global') -> Tensor:
    """Stack statistics and compute support also.

    Applies average strategy afterwards.

    """
    res = torch.stack([tp, fp, tn, fn, tp + fn], dim=-1)
    sum_dim = 0 if multidim_average == 'global' else 1
    if average == 'micro':
        return res.sum(sum_dim)
    if average == 'macro':
        return res.float().mean(sum_dim)
    if average == 'weighted':
        w = tp + fn
        return (res * (w / w.sum()).reshape(*w.shape, 1)).sum(sum_dim)
    if average is None or average == 'none':
        return res
    return None