from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.data import _bincount, select_topk
from torchmetrics.utilities.enums import AverageMethod, ClassificationTask, DataType, MDMCAverageMethod
def _multiclass_stat_scores_format(preds: Tensor, target: Tensor, top_k: int=1) -> Tuple[Tensor, Tensor]:
    """Convert all input to label format except if ``top_k`` is not 1.

    - Applies argmax if preds have one more dimension than target
    - Flattens additional dimensions

    """
    if preds.ndim == target.ndim + 1 and top_k == 1:
        preds = preds.argmax(dim=1)
    preds = preds.reshape(*preds.shape[:2], -1) if top_k != 1 else preds.reshape(preds.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    return (preds, target)