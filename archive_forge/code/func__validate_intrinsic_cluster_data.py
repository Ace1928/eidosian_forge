from typing import Optional, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
def _validate_intrinsic_cluster_data(data: Tensor, labels: Tensor) -> None:
    """Validate that the input data and labels have correct shape and type."""
    if data.ndim != 2:
        raise ValueError(f'Expected 2D data, got {data.ndim}D data instead')
    if not data.is_floating_point():
        raise ValueError(f'Expected floating point data, got {data.dtype} data instead')
    if labels.ndim != 1:
        raise ValueError(f'Expected 1D labels, got {labels.ndim}D labels instead')