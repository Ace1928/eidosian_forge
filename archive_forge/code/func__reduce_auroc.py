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
def _reduce_auroc(fpr: Union[Tensor, List[Tensor]], tpr: Union[Tensor, List[Tensor]], average: Optional[Literal['macro', 'weighted', 'none']]='macro', weights: Optional[Tensor]=None, direction: float=1.0) -> Tensor:
    """Reduce multiple average precision score into one number."""
    if isinstance(fpr, Tensor) and isinstance(tpr, Tensor):
        res = _auc_compute_without_check(fpr, tpr, direction=direction, axis=1)
    else:
        res = torch.stack([_auc_compute_without_check(x, y, direction=direction) for x, y in zip(fpr, tpr)])
    if average is None or average == 'none':
        return res
    if torch.isnan(res).any():
        rank_zero_warn(f'Average precision score for one or more classes was `nan`. Ignoring these classes in {average}-average', UserWarning)
    idx = ~torch.isnan(res)
    if average == 'macro':
        return res[idx].mean()
    if average == 'weighted' and weights is not None:
        weights = _safe_divide(weights[idx], weights[idx].sum())
        return (res[idx] * weights).sum()
    raise ValueError('Received an incompatible combinations of inputs to make reduction.')