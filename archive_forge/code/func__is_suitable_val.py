from typing import Any, Dict, Optional, Sequence, Union
import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.wrappers.abstract import WrapperMetric
@staticmethod
def _is_suitable_val(val: Union[float, Tensor]) -> bool:
    """Check whether min/max is a scalar value."""
    if isinstance(val, (int, float)):
        return True
    if isinstance(val, Tensor):
        return val.numel() == 1
    return False