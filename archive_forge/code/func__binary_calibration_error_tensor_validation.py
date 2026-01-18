from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
def _binary_calibration_error_tensor_validation(preds: Tensor, target: Tensor, ignore_index: Optional[int]=None) -> None:
    _binary_confusion_matrix_tensor_validation(preds, target, ignore_index)
    if not preds.is_floating_point():
        raise ValueError(f'Expected argument `preds` to be floating tensor with probabilities/logits but got tensor with dtype {preds.dtype}')