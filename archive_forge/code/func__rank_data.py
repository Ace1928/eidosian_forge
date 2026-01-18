from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
def _rank_data(data: Tensor) -> Tensor:
    """Calculate the rank for each element of a tensor.

    The rank refers to the indices of an element in the corresponding sorted tensor (starting from 1). Duplicates of the
    same value will be assigned the mean of their rank.

    Adopted from `Rank of element tensor`_

    """
    n = data.numel()
    rank = torch.empty_like(data)
    idx = data.argsort()
    rank[idx[:n]] = torch.arange(1, n + 1, dtype=data.dtype, device=data.device)
    repeats = _find_repeats(data)
    for r in repeats:
        condition = data == r
        rank[condition] = rank[condition].mean()
    return rank