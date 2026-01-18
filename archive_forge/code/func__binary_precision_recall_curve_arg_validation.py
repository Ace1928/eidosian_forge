from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide, interp
from torchmetrics.utilities.data import _bincount, _cumsum
from torchmetrics.utilities.enums import ClassificationTask
def _binary_precision_recall_curve_arg_validation(thresholds: Optional[Union[int, List[float], Tensor]]=None, ignore_index: Optional[int]=None) -> None:
    """Validate non tensor input.

    - ``threshold`` has to be None | a 1d tensor | a list of floats in the [0,1] range | an int
    - ``ignore_index`` has to be None or int

    """
    if thresholds is not None and (not isinstance(thresholds, (list, int, Tensor))):
        raise ValueError(f'Expected argument `thresholds` to either be an integer, list of floats or tensor of floats, but got {thresholds}')
    if isinstance(thresholds, int) and thresholds < 2:
        raise ValueError(f'If argument `thresholds` is an integer, expected it to be larger than 1, but got {thresholds}')
    if isinstance(thresholds, list) and (not all((isinstance(t, float) and 0 <= t <= 1 for t in thresholds))):
        raise ValueError(f'If argument `thresholds` is a list, expected all elements to be floats in the [0,1] range, but got {thresholds}')
    if isinstance(thresholds, Tensor) and (not thresholds.ndim == 1):
        raise ValueError('If argument `thresholds` is an tensor, expected the tensor to be 1d')
    if ignore_index is not None and (not isinstance(ignore_index, int)):
        raise ValueError(f'Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}')