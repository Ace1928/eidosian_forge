import inspect
import os
from typing import List, NamedTuple, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from typing_extensions import Literal
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_13
class _NoTrainLpips(_LPIPS):
    """Wrapper to make sure LPIPS never leaves evaluation mode."""

    def train(self, mode: bool) -> '_NoTrainLpips':
        """Force network to always be in evaluation mode."""
        return super().train(False)