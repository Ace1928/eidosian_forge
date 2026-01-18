from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.data import _bincount, select_topk
from torchmetrics.utilities.enums import AverageMethod, ClassificationTask, DataType, MDMCAverageMethod
def _binary_stat_scores_compute(tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor, multidim_average: Literal['global', 'samplewise']='global') -> Tensor:
    """Stack statistics and compute support also."""
    return torch.stack([tp, fp, tn, fn, tp + fn], dim=0 if multidim_average == 'global' else 1).squeeze()