import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torchmetrics.utilities.exceptions import TorchMetricsUserWarning
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_12, _TORCH_GREATER_EQUAL_1_13, _XLA_AVAILABLE
from torchmetrics.utilities.prints import rank_zero_warn
def _squeeze_scalar_element_tensor(x: Tensor) -> Tensor:
    return x.squeeze() if x.numel() == 1 else x