from copy import deepcopy
from typing import Any, List, Optional, Sequence, Tuple, Union
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch.nn import ModuleList
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.wrappers.abstract import WrapperMetric
def _get_nan_indices(*tensors: Tensor) -> Tensor:
    """Get indices of rows along dim 0 which have NaN values."""
    if len(tensors) == 0:
        raise ValueError('Must pass at least one tensor as argument')
    sentinel = tensors[0]
    nan_idxs = torch.zeros(len(sentinel), dtype=torch.bool, device=sentinel.device)
    for tensor in tensors:
        permuted_tensor = tensor.flatten(start_dim=1)
        nan_idxs |= torch.any(torch.isnan(permuted_tensor), dim=1)
    return nan_idxs