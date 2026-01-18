from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def _binary_confusion_matrix_arg_validation(threshold: float=0.5, ignore_index: Optional[int]=None, normalize: Optional[Literal['true', 'pred', 'all', 'none']]=None) -> None:
    """Validate non tensor input.

    - ``threshold`` has to be a float in the [0,1] range
    - ``ignore_index`` has to be None or int
    - ``normalize`` has to be "true" | "pred" | "all" | "none" | None

    """
    if not (isinstance(threshold, float) and 0 <= threshold <= 1):
        raise ValueError(f'Expected argument `threshold` to be a float in the [0,1] range, but got {threshold}.')
    if ignore_index is not None and (not isinstance(ignore_index, int)):
        raise ValueError(f'Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}')
    allowed_normalize = ('true', 'pred', 'all', 'none', None)
    if normalize not in allowed_normalize:
        raise ValueError(f'Expected argument `normalize` to be one of {allowed_normalize}, but got {normalize}.')