import inspect
import os
from typing import List, NamedTuple, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from typing_extensions import Literal
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_13
class _SqueezeOutput(NamedTuple):
    relu1: Tensor
    relu2: Tensor
    relu3: Tensor
    relu4: Tensor
    relu5: Tensor
    relu6: Tensor
    relu7: Tensor