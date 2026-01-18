from enum import Enum
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from xformers import _is_triton_available
from collections import namedtuple
class Residual(nn.Module, RequiresWrappedInputs):
    """
    Object-oriented handling of the residual path

    This supports scaling of the residual path, as proposed by DeepNet_
    .. _DeepNet: https://arxiv.org/pdf/2203.00555v1.pdf

    .. Note: the wrapped layers must accept all the inputs as a single list
    """

    def __init__(self, layer: nn.Module, scale: Optional[float]=None):
        super().__init__()
        self.layer = layer
        self.scale = scale
        self.wrap_inputs = isinstance(layer, RequiresWrappedInputs)

    def forward(self, inputs: List[torch.Tensor], **kwargs):
        if self.scale is not None:
            residue = inputs[0] * self.scale
        else:
            residue = inputs[0]
        if self.wrap_inputs:
            return residue + self.layer(inputs=inputs, **kwargs)
        else:
            return residue + self.layer(*inputs, **kwargs)