from enum import Enum
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from xformers import _is_triton_available
from collections import namedtuple
class PreNorm(nn.Module, RequiresWrappedInputs):
    """Adds a normalization before computing attention

    ..Note: If a list of inputs is passed, all of them get normalized"""

    def __init__(self, d_norm: int, sublayer: nn.Module, normalization: NormalizationType, use_triton: bool=True):
        super().__init__()
        if _is_triton_available() and use_triton and (normalization == NormalizationType.LayerNorm):
            self.norm: Union[nn.LayerNorm, FusedLayerNorm] = FusedLayerNorm(d_norm)
        else:
            self.norm = get_normalization_layer(normalization)(d_norm)
        self.sublayer = sublayer
        self.wrap_inputs = isinstance(sublayer, RequiresWrappedInputs)

    def forward(self, inputs: List[torch.Tensor], **kwargs):
        assert len(inputs) > 0
        ids = [id(x) for x in inputs]
        if ids.count(ids[0]) == len(ids):
            x_norm = self.norm(inputs[0])
            inputs_normed = [x_norm for _ in inputs]
        else:
            inputs_normed = [self.norm(x_) for x_ in inputs]
        if self.wrap_inputs:
            return self.sublayer(inputs=inputs_normed, **kwargs)
        else:
            return self.sublayer(*inputs_normed, **kwargs)