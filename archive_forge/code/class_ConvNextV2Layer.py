from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_convnextv2 import ConvNextV2Config
class ConvNextV2Layer(nn.Module):
    """This corresponds to the `Block` class in the original implementation.

    There are two equivalent implementations: [DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C,
    H, W) (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back

    The authors used (2) as they find it slightly faster in PyTorch.

    Args:
        config ([`ConvNextV2Config`]): Model configuration class.
        dim (`int`): Number of input channels.
        drop_path (`float`): Stochastic depth rate. Default: 0.0.
    """

    def __init__(self, config, dim, drop_path=0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.layernorm = ConvNextV2LayerNorm(dim, eps=1e-06)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = ACT2FN[config.hidden_act]
        self.grn = ConvNextV2GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = ConvNextV2DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        input = hidden_states
        x = self.dwconv(hidden_states)
        x = x.permute(0, 2, 3, 1)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x