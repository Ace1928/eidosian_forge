import itertools
from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import _multiclass_confusion_matrix_update
from torchmetrics.functional.nominal.utils import (
def _tschuprows_t_update(preds: Tensor, target: Tensor, num_classes: int, nan_strategy: Literal['replace', 'drop']='replace', nan_replace_value: Optional[float]=0.0) -> Tensor:
    """Compute the bins to update the confusion matrix with for Tschuprow's T calculation.

    Args:
        preds: 1D or 2D tensor of categorical (nominal) data
        target: 1D or 2D tensor of categorical (nominal) data
        num_classes: Integer specifying the number of classes
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN`s when ``nan_strategy = 'replace```

    Returns:
        Non-reduced confusion matrix

    """
    preds = preds.argmax(1) if preds.ndim == 2 else preds
    target = target.argmax(1) if target.ndim == 2 else target
    preds, target = _handle_nan_in_data(preds, target, nan_strategy, nan_replace_value)
    return _multiclass_confusion_matrix_update(preds, target, num_classes)