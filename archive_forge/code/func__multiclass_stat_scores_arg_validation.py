from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.data import _bincount, select_topk
from torchmetrics.utilities.enums import AverageMethod, ClassificationTask, DataType, MDMCAverageMethod
def _multiclass_stat_scores_arg_validation(num_classes: int, top_k: int=1, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='macro', multidim_average: Literal['global', 'samplewise']='global', ignore_index: Optional[int]=None) -> None:
    """Validate non tensor input.

    - ``num_classes`` has to be a int larger than 1
    - ``top_k`` has to be an int larger than 0 but no larger than number of classes
    - ``average`` has to be "micro" | "macro" | "weighted" | "none"
    - ``multidim_average`` has to be either "global" or "samplewise"
    - ``ignore_index`` has to be None or int

    """
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(f'Expected argument `num_classes` to be an integer larger than 1, but got {num_classes}')
    if not isinstance(top_k, int) and top_k < 1:
        raise ValueError(f'Expected argument `top_k` to be an integer larger than or equal to 1, but got {top_k}')
    if top_k > num_classes:
        raise ValueError(f'Expected argument `top_k` to be smaller or equal to `num_classes` but got {top_k} and {num_classes}')
    allowed_average = ('micro', 'macro', 'weighted', 'none', None)
    if average not in allowed_average:
        raise ValueError(f'Expected argument `average` to be one of {allowed_average}, but got {average}')
    allowed_multidim_average = ('global', 'samplewise')
    if multidim_average not in allowed_multidim_average:
        raise ValueError(f'Expected argument `multidim_average` to be one of {allowed_multidim_average}, but got {multidim_average}')
    if ignore_index is not None and (not isinstance(ignore_index, int)):
        raise ValueError(f'Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}')