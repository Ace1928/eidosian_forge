from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.data import _bincount, select_topk
from torchmetrics.utilities.enums import AverageMethod, ClassificationTask, DataType, MDMCAverageMethod
def _multilabel_stat_scores_arg_validation(num_labels: int, threshold: float=0.5, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='macro', multidim_average: Literal['global', 'samplewise']='global', ignore_index: Optional[int]=None) -> None:
    """Validate non tensor input.

    - ``num_labels`` should be an int larger than 1
    - ``threshold`` has to be a float in the [0,1] range
    - ``average`` has to be "micro" | "macro" | "weighted" | "none"
    - ``multidim_average`` has to be either "global" or "samplewise"
    - ``ignore_index`` has to be None or int

    """
    if not isinstance(num_labels, int) or num_labels < 2:
        raise ValueError(f'Expected argument `num_labels` to be an integer larger than 1, but got {num_labels}')
    if not (isinstance(threshold, float) and 0 <= threshold <= 1):
        raise ValueError(f'Expected argument `threshold` to be a float, but got {threshold}.')
    allowed_average = ('micro', 'macro', 'weighted', 'none', None)
    if average not in allowed_average:
        raise ValueError(f'Expected argument `average` to be one of {allowed_average}, but got {average}')
    allowed_multidim_average = ('global', 'samplewise')
    if multidim_average not in allowed_multidim_average:
        raise ValueError(f'Expected argument `multidim_average` to be one of {allowed_multidim_average}, but got {multidim_average}')
    if ignore_index is not None and (not isinstance(ignore_index, int)):
        raise ValueError(f'Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}')