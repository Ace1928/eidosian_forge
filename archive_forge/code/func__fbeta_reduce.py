from typing import Optional
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _adjust_weights_safe_divide, _safe_divide
from torchmetrics.utilities.enums import ClassificationTask
def _fbeta_reduce(tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor, beta: float, average: Optional[Literal['binary', 'micro', 'macro', 'weighted', 'none']], multidim_average: Literal['global', 'samplewise']='global', multilabel: bool=False) -> Tensor:
    beta2 = beta ** 2
    if average == 'binary':
        return _safe_divide((1 + beta2) * tp, (1 + beta2) * tp + beta2 * fn + fp)
    if average == 'micro':
        tp = tp.sum(dim=0 if multidim_average == 'global' else 1)
        fn = fn.sum(dim=0 if multidim_average == 'global' else 1)
        fp = fp.sum(dim=0 if multidim_average == 'global' else 1)
        return _safe_divide((1 + beta2) * tp, (1 + beta2) * tp + beta2 * fn + fp)
    fbeta_score = _safe_divide((1 + beta2) * tp, (1 + beta2) * tp + beta2 * fn + fp)
    return _adjust_weights_safe_divide(fbeta_score, average, multilabel, tp, fp, fn)