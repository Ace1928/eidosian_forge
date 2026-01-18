from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.data import _bincount, select_topk
from torchmetrics.utilities.enums import AverageMethod, ClassificationTask, DataType, MDMCAverageMethod
def _binary_stat_scores_arg_validation(threshold: float=0.5, multidim_average: Literal['global', 'samplewise']='global', ignore_index: Optional[int]=None) -> None:
    """Validate non tensor input.

    - ``threshold`` has to be a float in the [0,1] range
    - ``multidim_average`` has to be either "global" or "samplewise"
    - ``ignore_index`` has to be None or int

    """
    if not (isinstance(threshold, float) and 0 <= threshold <= 1):
        raise ValueError(f'Expected argument `threshold` to be a float in the [0,1] range, but got {threshold}.')
    allowed_multidim_average = ('global', 'samplewise')
    if multidim_average not in allowed_multidim_average:
        raise ValueError(f'Expected argument `multidim_average` to be one of {allowed_multidim_average}, but got {multidim_average}')
    if ignore_index is not None and (not isinstance(ignore_index, int)):
        raise ValueError(f'Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}')