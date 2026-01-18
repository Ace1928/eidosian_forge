import inspect
import os
from typing import List, NamedTuple, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from typing_extensions import Literal
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_13
class _VGGOutputs(NamedTuple):
    relu1_2: Tensor
    relu2_2: Tensor
    relu3_3: Tensor
    relu4_3: Tensor
    relu5_3: Tensor