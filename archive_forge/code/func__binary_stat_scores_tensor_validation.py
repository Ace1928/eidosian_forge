from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.data import _bincount, select_topk
from torchmetrics.utilities.enums import AverageMethod, ClassificationTask, DataType, MDMCAverageMethod
def _binary_stat_scores_tensor_validation(preds: Tensor, target: Tensor, multidim_average: Literal['global', 'samplewise']='global', ignore_index: Optional[int]=None) -> None:
    """Validate tensor input.

    - tensors have to be of same shape
    - all values in target tensor that are not ignored have to be in {0, 1}
    - if pred tensor is not floating point, then all values also have to be in {0, 1}
    - if ``multidim_average`` is set to ``samplewise`` preds tensor needs to be at least 2 dimensional

    """
    _check_same_shape(preds, target)
    unique_values = torch.unique(target)
    if ignore_index is None:
        check = torch.any((unique_values != 0) & (unique_values != 1))
    else:
        check = torch.any((unique_values != 0) & (unique_values != 1) & (unique_values != ignore_index))
    if check:
        raise RuntimeError(f'Detected the following values in `target`: {unique_values} but expected only the following values {([0, 1] if ignore_index is None else [ignore_index])}.')
    if not preds.is_floating_point():
        unique_values = torch.unique(preds)
        if torch.any((unique_values != 0) & (unique_values != 1)):
            raise RuntimeError(f'Detected the following values in `preds`: {unique_values} but expected only the following values [0,1] since `preds` is a label tensor.')
    if multidim_average != 'global' and preds.ndim < 2:
        raise ValueError('Expected input to be at least 2D when multidim_average is set to `samplewise`')