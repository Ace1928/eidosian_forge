from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def _multiclass_confusion_matrix_compute(confmat: Tensor, normalize: Optional[Literal['true', 'pred', 'all', 'none']]=None) -> Tensor:
    """Reduces the confusion matrix to it's final form.

    Normalization technique can be chosen by ``normalize``.

    """
    return _confusion_matrix_reduce(confmat, normalize)