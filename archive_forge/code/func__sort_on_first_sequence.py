from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount, _cumsum, dim_zero_cat
from torchmetrics.utilities.enums import EnumStr
def _sort_on_first_sequence(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    """Sort sequences in an ascent order according to the sequence ``x``."""
    y = torch.clone(y)
    x, y = (x.T, y.T)
    x, perm = x.sort()
    for i in range(x.shape[0]):
        y[i] = y[i][perm[i]]
    return (x.T, y.T)