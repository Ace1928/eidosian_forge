import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.distributed as dist
from torch import IntTensor, Tensor
from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import _cumsum
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _PYCOCOTOOLS_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class BaseMetricResults(dict):
    """Base metric class, that allows fields for pre-defined metrics."""

    def __getattr__(self, key: str) -> Tensor:
        """Get a specific metric attribute."""
        if key in self:
            return self[key]
        raise AttributeError(f'No such attribute: {key}')

    def __setattr__(self, key: str, value: Tensor) -> None:
        """Set a specific metric attribute."""
        self[key] = value

    def __delattr__(self, key: str) -> None:
        """Delete a specific metric attribute."""
        if key in self:
            del self[key]
        raise AttributeError(f'No such attribute: {key}')