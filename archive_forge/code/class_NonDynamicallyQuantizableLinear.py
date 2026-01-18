import math
from typing import Any
import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from .. import functional as F
from .. import init
from .module import Module
from .lazy import LazyModuleMixin
class NonDynamicallyQuantizableLinear(Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool=True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)