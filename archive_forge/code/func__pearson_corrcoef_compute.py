import math
from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _check_same_shape
def _pearson_corrcoef_compute(var_x: Tensor, var_y: Tensor, corr_xy: Tensor, nb: Tensor) -> Tensor:
    """Compute the final pearson correlation based on accumulated statistics.

    Args:
        var_x: variance estimate of x tensor
        var_y: variance estimate of y tensor
        corr_xy: covariance estimate between x and y tensor
        nb: number of observations

    """
    var_x /= nb - 1
    var_y /= nb - 1
    corr_xy /= nb - 1
    if var_x.dtype == torch.float16 and var_x.device == torch.device('cpu'):
        var_x = var_x.bfloat16()
        var_y = var_y.bfloat16()
    bound = math.sqrt(torch.finfo(var_x.dtype).eps)
    if (var_x < bound).any() or (var_y < bound).any():
        rank_zero_warn(f'The variance of predictions or target is close to zero. This can cause instability in Pearson correlationcoefficient, leading to wrong results. Consider re-scaling the input if possible or computing using alarger dtype (currently using {var_x.dtype}).', UserWarning)
    corrcoef = (corr_xy / (var_x * var_y).sqrt()).squeeze()
    return torch.clamp(corrcoef, -1.0, 1.0)